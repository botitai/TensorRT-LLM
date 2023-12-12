import argparse
import json
import time
from pathlib import Path

# isort: off
import torch
import tensorrt as trt
# isort: on
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._utils import trt_dtype_to_torch
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from build import get_engine_name  # isort:skip


def print_tensor(tensor_name, tensor, num_elements=10):
    print(
        f'{tensor_name}: mean={tensor.abs().mean().item():.3f}, sum={tensor.abs().sum().item():.3f}, max={tensor.abs().max().item():.3f}'
    )
    # Pass num_elements=-1 will print the whole tensor
    if num_elements < 0:
        num_elements = torch.numel(tensor)
    print(f'{tensor.flatten()[:num_elements]}')
    print("Tensor Shape: ", tensor.size())
    print("")


def read_config(config_path: Path):
    with open(config_path, "r") as f:
        config = json.load(f)
    use_gpt_attention_plugin = config["plugin_config"]["gpt_attention_plugin"]
    remove_input_padding = config["plugin_config"]["remove_input_padding"]
    tp_size = config['builder_config']['tensor_parallel']
    pp_size = config['builder_config']['pipeline_parallel']
    gpus_per_node = config['builder_config']['gpus_per_node']
    world_size = tp_size * pp_size
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config["builder_config"]["num_heads"]
    hidden_size = config["builder_config"]["hidden_size"]
    head_size = config["builder_config"]["head_size"]
    vocab_size = config["builder_config"]["vocab_size"]
    num_layers = config["builder_config"]["num_layers"]
    num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)

    assert (num_heads % tp_size) == 0
    num_heads = num_heads // tp_size
    hidden_size = hidden_size // tp_size
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

    cross_attention = config["builder_config"]["cross_attention"]
    has_position_embedding = config["builder_config"]["has_position_embedding"]
    has_token_type_embedding = config["builder_config"][
        "has_token_type_embedding"]
    use_custom_all_reduce = config['plugin_config'].get('use_custom_all_reduce',
                                                        False)
    dtype = config["builder_config"]["precision"]

    model_config = ModelConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        head_size=head_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        cross_attention=cross_attention,
        has_position_embedding=has_position_embedding,
        has_token_type_embedding=has_token_type_embedding,
        use_custom_all_reduce=use_custom_all_reduce,
        dtype=dtype,
        gather_all_token_logits=True)

    return model_config, tp_size, pp_size, gpus_per_node, dtype


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument('--max_kv_cache_len',
                        type=int,
                        default=None,
                        help='The max kv cache length. \
              If the final sequence length exceeds the kv cache length, we will enable cyclic kv cache. \
              If it is set to None, we will use the max sequence length.')
    parser.add_argument("--log_level", type=str, default="error")
    parser.add_argument("--engine_dir", "-i", type=str, default="trt_engines")
    parser.add_argument("--engine_name", type=str, default="enc_dec")
    parser.add_argument("--model_name",
                        type=str,
                        help="HuggingFace model name",
                        default="t5-small")
    parser.add_argument("--num_beams",
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument("--debug_mode",
                        help="Whether or not to turn on the debug mode",
                        action='store_true')
    parser.add_argument("--compare_hf_fp32",
                        help="Compare results with HuggingFace FP32",
                        action='store_true')
    return parser.parse_args()


class TRTLLMEncDecModel:

    def __init__(self, engine_name, engine_dir, debug_mode=False):
        # in multi-node setup, it's important to set_device at the very beginning so .to('cuda') refers to current device
        # accordingly, all input & output tensors should be moved to current device
        # otherwise, it's default to 'cuda:0'
        self.runtime_rank = tensorrt_llm.mpi_rank()
        device_id = self.runtime_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        self.device = torch.cuda.current_device()

        engine_dir = Path(engine_dir)

        def engine_setup(component):
            # model config
            config_path = engine_dir / component / "config.json"
            model_config, tp_size, pp_size, gpus_per_node, dtype = read_config(
                config_path)

            # MGMN config
            world_size = tp_size * pp_size
            runtime_rank = tensorrt_llm.mpi_rank()
            assert runtime_rank < world_size, "Runtime GPU rank exceeds MPI world size. Did you launch more MPI processes than required?"
            runtime_mapping = tensorrt_llm.Mapping(world_size,
                                                   runtime_rank,
                                                   tp_size=tp_size,
                                                   pp_size=pp_size,
                                                   gpus_per_node=gpus_per_node)

            # load engine
            engine_fname = get_engine_name(engine_name, dtype, tp_size, pp_size,
                                           runtime_rank)
            with open(engine_dir / component / engine_fname, "rb") as f:
                engine_buffer = f.read()

            return model_config, runtime_mapping, engine_buffer

        # Note: encoder and decoder doesn't necessarily have the same TP & PP config
        self.encoder_model_config, self.encoder_runtime_mapping, encoder_engine_buffer = engine_setup(
            component='encoder')
        self.decoder_model_config, self.decoder_runtime_mapping, decoder_engine_buffer = engine_setup(
            component='decoder')

        # for Pipeline Parallelism in encoder
        self.nccl_comm = torch.classes.FasterTransformer.NcclCommunicatorOp(
            self.encoder_runtime_mapping.tp_size,
            self.encoder_runtime_mapping.pp_size,
            self.encoder_runtime_mapping.rank)

        # session setup
        self.encoder_session = tensorrt_llm.runtime.Session.from_serialized_engine(
            encoder_engine_buffer)
        self.decoder_session = tensorrt_llm.runtime.GenerationSession(
            self.decoder_model_config,
            decoder_engine_buffer,
            self.decoder_runtime_mapping,
            debug_mode=debug_mode)
        self.stream = torch.cuda.current_stream().cuda_stream

    @classmethod
    def from_engine(cls, engine_name, engine_dir, debug_mode=False):
        return cls(engine_name, engine_dir, debug_mode=debug_mode)

    def process_input(self,
                      input_ids,
                      remove_input_padding=False,
                      pad_token_id=0):
        if remove_input_padding:
            # in remove padding mode --> flatten input, calculate actual length and max length
            # Note: 1st token should never be removed, even if it is pad_token_id
            first_ids = input_ids[:, 0]
            input_ids = input_ids[:, 1:]
            input_lengths = 1 + (input_ids != pad_token_id).sum(dim=1).type(
                torch.IntTensor).to(self.device)  # [batch_size]
            new_ids = []
            for i in range(len(input_ids)):
                row = input_ids[i, :]
                row = row[row != pad_token_id]
                new_ids.append(
                    torch.cat(
                        (torch.IntTensor([first_ids[i]]).to(self.device), row)))
            input_ids = torch.cat(new_ids).unsqueeze(dim=0)  # [1, num_tokens]
        else:
            # in padding mode --> keep input, just calculate actual length and max length
            # Note: 1st token should always count, even if it is pad_token_id. e.g., decoder start id in enc-dec models could be a single pad_token_id, we should count
            input_lengths = torch.tensor(
                1 + (input_ids[:, 1:] != pad_token_id).sum(dim=1).type(
                    torch.IntTensor).to(self.device),
                dtype=torch.int32,
                device=self.device)
        max_input_length = torch.max(input_lengths).item()
        return input_ids, input_lengths, max_input_length

    def encoder_run(self,
                    input_ids,
                    input_lengths,
                    max_input_length,
                    position_ids=None,
                    token_type_ids=None,
                    debug_mode=False):

        # each engine has hidden_dim/TP, don't forget to multiply TP
        hidden_size = self.encoder_model_config.hidden_size * self.encoder_runtime_mapping.tp_size
        hidden_states_shape = (input_ids.shape[0], input_ids.shape[1],
                               hidden_size)  # [1,num_tokens,D] or [BS,seqlen,D]
        hidden_states_dtype = lambda name: trt_dtype_to_torch(
            self.encoder_session.engine.get_tensor_dtype(name))

        # input tensors. only first PP rank has id input, others are hidden_states input
        inputs = {}
        if self.encoder_runtime_mapping.is_first_pp_rank():
            inputs['input_ids'] = input_ids.contiguous()
            if self.encoder_model_config.has_position_embedding:
                inputs['position_ids'] = position_ids.contiguous()
            if self.encoder_model_config.has_token_type_embedding:
                inputs['token_type_ids'] = token_type_ids.contiguous()
        else:
            # just need a placeholder, engine will call NCCL to recv and fill data from previous rank
            inputs['hidden_states_input'] = torch.empty(
                hidden_states_shape,
                dtype=hidden_states_dtype('hidden_states_input'),
                device=self.device).contiguous()
        inputs['input_lengths'] = input_lengths
        # use shape info to pass max length info in remove padding mode
        inputs['max_input_length'] = torch.empty(
            (max_input_length, ),
            dtype=hidden_states_dtype('max_input_length'),
            device=self.device).contiguous()

        # output tensors. only last PP rank final encoder output, others are intermediate hidden_states output. Need broadcast later
        outputs = {}
        if self.encoder_runtime_mapping.is_last_pp_rank():
            outputs['encoder_output'] = torch.empty(
                hidden_states_shape,
                dtype=hidden_states_dtype('encoder_output'),
                device=self.device).contiguous()
        else:
            outputs['hidden_states_output'] = torch.empty(
                hidden_states_shape,
                dtype=hidden_states_dtype('hidden_states_output'),
                device=self.device).contiguous()

        # -------------------------------------------
        if debug_mode:
            engine = self.encoder_session.engine
            context = self.encoder_session.context
            # setup debugging buffer for the encoder
            for i in range(self.encoder_session.engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(
                        name
                ) == trt.TensorIOMode.OUTPUT and name not in outputs.keys():
                    dtype = engine.get_tensor_dtype(name)
                    shape = context.get_tensor_shape(name)
                    outputs[name] = torch.zeros(tuple(shape),
                                                dtype=trt_dtype_to_torch(dtype),
                                                device=self.device)
                    context.set_tensor_address(name, outputs[name].data_ptr())
        # -------------------------------------------

        # TRT session run
        # Note: runtime.Session's run() method will set input/output tensor address, here we only need to provide tensor shape
        self.encoder_session.set_shapes(inputs)
        ok = self.encoder_session.run(inputs, outputs, self.stream)
        assert ok, "Runtime execution failed"
        torch.cuda.synchronize()

        # Tensor Parallelism is handled by model/engine definition
        # But we need to broadcast among PP group at the end of encoder's Pipeline Parallelism
        # After this, all ranks should recv the encoder output, and world might be re-configured using decoder's TP-PP config
        def pp_communicate_encoder_output(encoder_output):
            if self.encoder_runtime_mapping.is_last_pp_rank():
                for pp_rank in self.encoder_runtime_mapping.pp_group:
                    if pp_rank != self.encoder_runtime_mapping.rank:
                        self.nccl_comm.send(encoder_output, pp_rank)
                return encoder_output
            else:
                self.nccl_comm.recv(encoder_output,
                                    self.encoder_runtime_mapping.pp_group[-1])
                return encoder_output

        if self.encoder_runtime_mapping.has_pp():
            # use hidden_states output buffer to receive output as the shapes are same
            encoder_output_buf = outputs[
                'encoder_output'] if self.encoder_runtime_mapping.is_last_pp_rank(
                ) else outputs['hidden_states_output']
            encoder_output = pp_communicate_encoder_output(encoder_output_buf)
        else:
            encoder_output = outputs['encoder_output']

        # -------------------------------------------
        if debug_mode:
            torch.cuda.synchronize()
            # use print_tensor() to print the tensors registered in the encoder network
            print("--------------------------------------")
            print("Debug output for Encoder")
            print("--------------------------------------")
            print("Registered output tensors are: ", outputs.keys())
            print_tensor('encoder_output', encoder_output)
            print("--------------------------------------")
        # -------------------------------------------

        return encoder_output

    def generate(
        self,
        encoder_input_ids,
        decoder_input_ids,
        max_new_tokens,
        num_beams=1,
        pad_token_id=None,
        eos_token_id=None,
        bos_token_id=None,
        debug_mode=False,
    ):
        ## ensure all externally provided tensors are on the correct device.
        encoder_input_ids = encoder_input_ids.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)

        ## encoder run
        logger.info(f"Rank {self.runtime_rank} Running encoder engine ...")
        encoder_input_ids, encoder_input_lengths, encoder_max_input_length = self.process_input(
            encoder_input_ids, self.encoder_model_config.remove_input_padding,
            pad_token_id)
        encoder_output = self.encoder_run(encoder_input_ids,
                                          encoder_input_lengths,
                                          encoder_max_input_length,
                                          debug_mode=debug_mode)

        ## decoder run
        logger.info(f"Rank {self.runtime_rank} Running decoder engine ...")
        decoder_input_ids, decoder_input_lengths, decoder_max_input_length = self.process_input(
            decoder_input_ids, self.decoder_model_config.remove_input_padding,
            pad_token_id)

        # generation config
        sampling_config = SamplingConfig(end_id=eos_token_id,
                                         pad_id=pad_token_id,
                                         num_beams=num_beams,
                                         min_length=1)

        # decoder autoregressive generation
        self.decoder_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            num_beams,
            max_kv_cache_length=None,
            encoder_max_input_length=encoder_max_input_length)
        torch.cuda.synchronize()

        output_ids = self.decoder_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_output,
            encoder_input_lengths=encoder_input_lengths,
            return_dict=True,
        )
        torch.cuda.synchronize()

        return output_ids


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_arguments()
    logger.set_level(args.log_level)

    test_remove_padding = True
    if not test_remove_padding:
        input_text = "translate English to German: The house is wonderful, radiating timeless charm and offering a warm, inviting interior with beautiful details and a serene backyard."
    else:
        input_text = [
            # "translate English to German: The house is wonderful.",
            # "summarize: I am a high-performance inference optimizer and runtime.",
            # "During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world",
            # "question: Is Earth sphere?",
            # "question: Is Earth flat?",
            "question: Is this response for the given user query consistent with the given premise? </s> response: Trailing return is a measure of the performance of an investment over a specific period of time. It is calculated by measuring the percentage price change of the investment over the specified time period, assuming that all dividends and distributions are reinvested. For example, if you invested $100 in a mutual fund on January 1st and the value of your investment grew to $110 by June 30th, your trailing return for the six-month period would be 10%. Would you like to know more about mutual fund performance metrics? </s> user query: User wants a more detailed explanation of trailing return. </s> premise: Title: Aditya Birla Sun Life India GenNext Fund | ABSL India GenNext Fund - ABSLMF\n\nSection: Value Added Products:\n\n### Withdrawal Frequency:\n\nFor Fixed Withdrawal Option: Investors can withdraw fixed amount on 1 or 7 or 10 or 14 or 20 or 21 or 28 of month/quarter/Half yearly and Annually for minimum 6 months/ 4 quarter/2 half years and 1 year.\nFor Appreciation Withdrawal Option: Investors can withdraw appreciation on the 1 of each month/quarter for minimum 6 months/ 4 quarter.\n\nCATP\n\nCapital Appreciation Transfer Plan (CATP) allows investors to preserve their capital and transfer only capital appreciation to another asset class / scheme at regular intervals\n\nTitle: Aditya Birla Sun Life MNC Fund |ABSL MNC Fund - ABSLMF\n\nSection: Value Added Products:\n\n### Withdrawal Frequency:\n\nFor Fixed Withdrawal Option: Investors can withdraw fixed amount on 1 or 7 or 10 or 14 or 20 or 21 or 28 of month/quarter/Half yearly and Annually for minimum 6 months/ 4 quarter/2 half years and 1 year.\n\nFor Appreciation Withdrawal Option: Investors can withdraw appreciation on the 1 of each month/quarter for minimum 6 months/ 4 quarter.\n\nCATP\n\nCapital Appreciation Transfer Plan (CATP) allows investors to preserve their capital and transfer only capital appreciation to another asset class / scheme at regular intervals\n\nTitle: Aditya Birla Sun Life India GenNext Fund | ABSL India GenNext Fund - ABSLMF\n\nSection: Value Added Products:\n\n### Minimum Transfer Amount\n\nFor STP installments greater than Rs.500 but less than Rs.999, Investors are required to instruct for minimum 12 transfers of Rs.500 and in multiples of Re. 1thereafter.\nFor STP installments of Rs.1000 and above, Investors are required to instruct for minimum 6 transfers of Rs.1000 and in multiples of Re. 1 thereafter.\n\nSWP\n\nSystematic Withdrawal Plan allows investors to withdraw a fixed amount of money from their mutual fund to build sustainable income streams while saving on Tax also\n\nInvestors have the option of:\n\nFixed Withdrawal\n- which allows investors of the Growth Plan to withdraw a fixed amount at regular intervals. Investors can withdraw fixed amount of Rs1,000/- each and above at regular intervals.\n\nAppreciation Withdrawal - which allows investors of Growth Plan to withdraw the appreciation amount at regular intervals. Investors can withdraw appreciation of Rs1,000/- and above at regular intervals.\n\nTitle: Aditya Birla Sun Life MNC Fund |ABSL MNC Fund - ABSLMF\n\nSection: Value Added Products:\n\n### Minimum Transfer Amount\n\nFor STP installments greater than Rs.500 but less than Rs.999, Investors are required to instruct for minimum 12 transfers of Rs.500 and in multiples of Re. 1thereafter.\n\nFor STP installments of Rs.1000 and above, Investors are required to instruct for minimum 6 transfers of Rs.1000 and in multiples of Re. 1 thereafter.\n\nSWP\n\nSystematic Withdrawal Plan allows investors to withdraw a fixed amount of money from their mutual fund to build sustainable income streams while saving on Tax also\n\nInvestors have the option of:\n\nFixed Withdrawal\n\n- which allows investors of the Growth Plan to withdraw a fixed amount at regular intervals. Investors can withdraw fixed amount of Rs1,000/- each and above at regular intervals.\n\nAppreciation Withdrawal - which allows investors of Growth Plan to withdraw the appreciation amount at regular intervals. Investors can withdraw appreciation of Rs1,000/- and above at regular intervals.\n\nTitle: Aditya Birla Sun Life Business Cycle Fund | Business Cycle Fund - ABSLMF\n\nSection: Fund Management\n\nAsit C Mehta Investment Intermediates Ltd.\n\nView Fund Managed by\n\nMr. Anil Shah\nMr. Dhaval Gala\nMr. Dhaval Joshi\n\n| Fund | 1 year3 years5 yearsSince Inception |\n| --- | --- |\n| Fund | 1 year | 3 years | 5 years | Since Inception |\n\nLoad More\n\nTotal Schemes managed by {fundmanagername} is {fundmanagerfundcount}\n\nDifferent plans shall have a different expense structure. The performance details provided herein are of (regular / direct) plan.\n\nTitle: Aditya Birla Sun Life Banking & Financial Services Fund - Regular Plan - ABSLMF\n\nSection: Highlights\n\n### Banking on New Opportunities\n\nFortunes of the banking and financial services sector are typically linked with economic growth. There are numerous factors that work in favour of the banking and financial services sector. Some of the key factors are: robust demand from middle class, rural penetration and technology-enabled services. According to a report by the National Council for Applied Economic Research's (NCAER) Centre for Macro Consumer Research, by 2015-16, India will be a country of 53.3 million middle class households, translating into 267 million people. Characteristics of the rising middle class include higher purchasing power and also the ability take on extra debt to meet their aspiring lifestyle. Similarly with the advent of technology, the reach of banks has extended to envelope the rural population that was previously unbankable. As a result, the banking and financial services sector has been able to deliver better returns. In this backdrop, Aditya Birla Sun Life Banking and Financial Services Fund (ABSL Banking and Financial Services Fund) could be your answer to unlocking higher returns.\n\nTitle: Aditya Birla Sun Life Dividend Yield Plus Fund |Dividend Yield Plus Fund - ABSLMF\n\nSection: Highlights\n\n### Get best of both worlds but with lesser risk\n\nAs the companies grow and exhibit their financial progress with steady growth and profit sharing through dividend distribution, they have steady share price appreciation and face lesser price volatility. High dividend payout often signals that there is enough cash generation in the business and that the stock might be under-priced given the high cash generating ability of the issuer. This provides investors an opportunity to earn regular dividend and good capital appreciation with lesser risk.",
            "explain: Is this response for the given user query consistent with the given premise? </s> response: Trailing return is a measure of the performance of an investment over a specific period of time. It is calculated by measuring the percentage price change of the investment over the specified time period, assuming that all dividends and distributions are reinvested. For example, if you invested $100 in a mutual fund on January 1st and the value of your investment grew to $110 by June 30th, your trailing return for the six-month period would be 10%. Would you like to know more about mutual fund performance metrics? </s> user query: User wants a more detailed explanation of trailing return. </s> premise: Title: Aditya Birla Sun Life India GenNext Fund | ABSL India GenNext Fund - ABSLMF\n\nSection: Value Added Products:\n\n### Withdrawal Frequency:\n\nFor Fixed Withdrawal Option: Investors can withdraw fixed amount on 1 or 7 or 10 or 14 or 20 or 21 or 28 of month/quarter/Half yearly and Annually for minimum 6 months/ 4 quarter/2 half years and 1 year.\nFor Appreciation Withdrawal Option: Investors can withdraw appreciation on the 1 of each month/quarter for minimum 6 months/ 4 quarter.\n\nCATP\n\nCapital Appreciation Transfer Plan (CATP) allows investors to preserve their capital and transfer only capital appreciation to another asset class / scheme at regular intervals\n\nTitle: Aditya Birla Sun Life MNC Fund |ABSL MNC Fund - ABSLMF\n\nSection: Value Added Products:\n\n### Withdrawal Frequency:\n\nFor Fixed Withdrawal Option: Investors can withdraw fixed amount on 1 or 7 or 10 or 14 or 20 or 21 or 28 of month/quarter/Half yearly and Annually for minimum 6 months/ 4 quarter/2 half years and 1 year.\n\nFor Appreciation Withdrawal Option: Investors can withdraw appreciation on the 1 of each month/quarter for minimum 6 months/ 4 quarter.\n\nCATP\n\nCapital Appreciation Transfer Plan (CATP) allows investors to preserve their capital and transfer only capital appreciation to another asset class / scheme at regular intervals\n\nTitle: Aditya Birla Sun Life India GenNext Fund | ABSL India GenNext Fund - ABSLMF\n\nSection: Value Added Products:\n\n### Minimum Transfer Amount\n\nFor STP installments greater than Rs.500 but less than Rs.999, Investors are required to instruct for minimum 12 transfers of Rs.500 and in multiples of Re. 1thereafter.\nFor STP installments of Rs.1000 and above, Investors are required to instruct for minimum 6 transfers of Rs.1000 and in multiples of Re. 1 thereafter.\n\nSWP\n\nSystematic Withdrawal Plan allows investors to withdraw a fixed amount of money from their mutual fund to build sustainable income streams while saving on Tax also\n\nInvestors have the option of:\n\nFixed Withdrawal\n- which allows investors of the Growth Plan to withdraw a fixed amount at regular intervals. Investors can withdraw fixed amount of Rs1,000/- each and above at regular intervals.\n\nAppreciation Withdrawal - which allows investors of Growth Plan to withdraw the appreciation amount at regular intervals. Investors can withdraw appreciation of Rs1,000/- and above at regular intervals.\n\nTitle: Aditya Birla Sun Life MNC Fund |ABSL MNC Fund - ABSLMF\n\nSection: Value Added Products:\n\n### Minimum Transfer Amount\n\nFor STP installments greater than Rs.500 but less than Rs.999, Investors are required to instruct for minimum 12 transfers of Rs.500 and in multiples of Re. 1thereafter.\n\nFor STP installments of Rs.1000 and above, Investors are required to instruct for minimum 6 transfers of Rs.1000 and in multiples of Re. 1 thereafter.\n\nSWP\n\nSystematic Withdrawal Plan allows investors to withdraw a fixed amount of money from their mutual fund to build sustainable income streams while saving on Tax also\n\nInvestors have the option of:\n\nFixed Withdrawal\n\n- which allows investors of the Growth Plan to withdraw a fixed amount at regular intervals. Investors can withdraw fixed amount of Rs1,000/- each and above at regular intervals.\n\nAppreciation Withdrawal - which allows investors of Growth Plan to withdraw the appreciation amount at regular intervals. Investors can withdraw appreciation of Rs1,000/- and above at regular intervals.\n\nTitle: Aditya Birla Sun Life Business Cycle Fund | Business Cycle Fund - ABSLMF\n\nSection: Fund Management\n\nAsit C Mehta Investment Intermediates Ltd.\n\nView Fund Managed by\n\nMr. Anil Shah\nMr. Dhaval Gala\nMr. Dhaval Joshi\n\n| Fund | 1 year3 years5 yearsSince Inception |\n| --- | --- |\n| Fund | 1 year | 3 years | 5 years | Since Inception |\n\nLoad More\n\nTotal Schemes managed by {fundmanagername} is {fundmanagerfundcount}\n\nDifferent plans shall have a different expense structure. The performance details provided herein are of (regular / direct) plan.\n\nTitle: Aditya Birla Sun Life Banking & Financial Services Fund - Regular Plan - ABSLMF\n\nSection: Highlights\n\n### Banking on New Opportunities\n\nFortunes of the banking and financial services sector are typically linked with economic growth. There are numerous factors that work in favour of the banking and financial services sector. Some of the key factors are: robust demand from middle class, rural penetration and technology-enabled services. According to a report by the National Council for Applied Economic Research's (NCAER) Centre for Macro Consumer Research, by 2015-16, India will be a country of 53.3 million middle class households, translating into 267 million people. Characteristics of the rising middle class include higher purchasing power and also the ability take on extra debt to meet their aspiring lifestyle. Similarly with the advent of technology, the reach of banks has extended to envelope the rural population that was previously unbankable. As a result, the banking and financial services sector has been able to deliver better returns. In this backdrop, Aditya Birla Sun Life Banking and Financial Services Fund (ABSL Banking and Financial Services Fund) could be your answer to unlocking higher returns.\n\nTitle: Aditya Birla Sun Life Dividend Yield Plus Fund |Dividend Yield Plus Fund - ABSLMF\n\nSection: Highlights\n\n### Get best of both worlds but with lesser risk\n\nAs the companies grow and exhibit their financial progress with steady growth and profit sharing through dividend distribution, they have steady share price appreciation and face lesser price volatility. High dividend payout often signals that there is enough cash generation in the business and that the stock might be under-priced given the high cash generating ability of the issuer. This provides investors an opportunity to earn regular dividend and good capital appreciation with lesser risk.",
            "question: Is this response for the given user query consistent with the given premise? </s> response: Aditya Birla Sun Life Pure Value Fund is an open-ended equity scheme that follows a value investment strategy. The objective of the scheme is to generate consistent long-term capital appreciation through value investing strategy. The fund invests in stocks available at discounted or lower price compared to prices based on stocks fundamentals. This product is suitable for investors who are seeking long term capital growth and investment predominantly in equity and equity-related securities following value investing strategy. The minimum investment is Rs. 1000/- and the latest NAV is 79.914 (as on 08-Jun-2023). The fund is managed by Mr. Kunal Sangoi & Mr. Dhaval Joshi. Would you like to know more about the fund? </s> user query: User wants to know about the fund summary of Aditya Birla Sun Life Pure Value Fund. </s> premise: Title: Aditya Birla Sun Life Pure Value Fund |ABSL Pure Value Fund - ABSLMF\n\nSection: Highlights\n\n### Aditya Birla Sun Life Pure Value Fund\n\n(An open ended equity scheme following a value investment strategy)\n\nTitle: Aditya Birla Sun Life Pure Value Fund |ABSL Pure Value Fund - ABSLMF\n\nSection: Highlights\n\n### Works for investors seeking higher long term returns\n\nEvery investor will like to buy stocks at low prices and sell them at high prices. But thats easier said than done since it requires investment acumen. It is this expertise that you get when you invest in Aditya Birla Sun Life Pure Value Fund. It follows value investment strategy i.e. the fund invests in the stocks available at discounted or lower price compared to prices based on stocks fundamentals.\n\nTitle: Aditya Birla Sun Life Pure Value Fund |ABSL Pure Value Fund - ABSLMF\n\nSection: Investment Objective\n\nThe Objective of the scheme is to generate consistent long-term capital appreciation through value investing strategy.\n\nTitle: Aditya Birla Sun Life Pure Value Fund |ABSL Pure Value Fund - ABSLMF\n\nSection: Highlights\n\n### This product is suitable for investors who are seeking\\*\n* Long term capital growth\n* Investment predominantly in equity and equity-related securities following value investing strategy i.e. investing in stocks priced lower than their true value, based on fundamentals\n\n\\*Investors should consult their financial advisers if in doubt whether the product is suitable for them\n\n<img alt=\"Moderately High Risk\" src=\"https://mutualfund.adityabirlacapital.com/wealth-creation-solutions/abc5_files/ADITYA-BIRLA-SUN-LIFE-PURE-VALUE-FUND.jpg\"/>\n\n[Invest Now](https://mutualfund.adityabirlacapital.com/portal/investor/quick-invest/fund-select;fundcode=PVF?null)\n[Add to Wishlist](https://mutualfund.adityabirlacapital.com/wealth-creation-solutions/None)\n\nAdd to Compare\n\nShow More\n\nTitle: Aditya Birla Sun Life Pure Value Fund |ABSL Pure Value Fund - ABSLMF\n\nSection: Fund Details\n* Fund Type - Open-Ended\n* Category - Equity\n* Sub-Category - Value Fund\n* Min Investment - Rs. 1000/-\n* Fund Manager - Mr. Kunal Sangoi & Mr. Dhaval Joshi\n* Latest NAV - 79.914 (as on 08-Jun-2023)\n* Inception Date - Mar 28, 2008\n\nView Full Details\n\nTitle: Aditya Birla Sun Life Pure Value Fund |ABSL Pure Value Fund - ABSLMF\n\nSection: This product is suitable for investors who are seeking\n* Long term capital growth\n* Investment predominantly in equity and equity-related securities following value investing strategy i.e. investing in stocks priced lower than their true value, based on fundamentals\n\nTitle: Aditya Birla Sun Life Pure Value Fund |ABSL Pure Value Fund - ABSLMF\n\nSection: Fund Management\n\n| [Aditya Birla Sun Life Pure Value Fund](https://mutualfund.adityabirlacapital.com/wealth-creation-solutions/aditya-birla-sun-life-pure-value-fund) | 22.38 % | 27.71 % | 6.13 % | 14.64 % |\n| [Aditya Birla Sun Life Digital India Fund](https://mutualfund.adityabirlacapital.com/most-popular-funds/aditya-birla-sun-life-digital-india-fund) | 5.65 % | 31.82 % | 19.54 % | 11.27 % |\n| [Aditya Birla Sun Life Focused Equity Fund](https://mutualfund.adityabirlacapital.com/most-popular-funds/aditya-birla-sun-life-focused-equity-fund) | 13.48 % | 21.47 % | 11.01 % | 13.67 % |\n| [Aditya Birla Sun Life Digital India Fund](https://mutualfund.adityabirlacapital.com/our-best-sip-fund/aditya-birla-sun-life-digital-india-fund) | 5.65 % | 31.82 % | 19.54 % | 11.27 % |\n\nLoad More\n\nTotal Schemes managed by Mr. Kunal Sangoi is 6\n\nDifferent plans shall have a different expense structure. The performance details provided herein are of (regular / direct) plan.",
            "explain: Is this response for the given user query consistent with the given premise? </s> response: Aditya Birla Sun Life Pure Value Fund is an open-ended equity scheme that follows a value investment strategy. The objective of the scheme is to generate consistent long-term capital appreciation through value investing strategy. The fund invests in stocks available at discounted or lower price compared to prices based on stocks fundamentals. This product is suitable for investors who are seeking long term capital growth and investment predominantly in equity and equity-related securities following value investing strategy. The minimum investment is Rs. 1000/- and the latest NAV is 79.914 (as on 08-Jun-2023). The fund is managed by Mr. Kunal Sangoi & Mr. Dhaval Joshi. Would you like to know more about the fund? </s> user query: User wants to know about the fund summary of Aditya Birla Sun Life Pure Value Fund. </s> premise: Title: Aditya Birla Sun Life Pure Value Fund |ABSL Pure Value Fund - ABSLMF\n\nSection: Highlights\n\n### Aditya Birla Sun Life Pure Value Fund\n\n(An open ended equity scheme following a value investment strategy)\n\nTitle: Aditya Birla Sun Life Pure Value Fund |ABSL Pure Value Fund - ABSLMF\n\nSection: Highlights\n\n### Works for investors seeking higher long term returns\n\nEvery investor will like to buy stocks at low prices and sell them at high prices. But thats easier said than done since it requires investment acumen. It is this expertise that you get when you invest in Aditya Birla Sun Life Pure Value Fund. It follows value investment strategy i.e. the fund invests in the stocks available at discounted or lower price compared to prices based on stocks fundamentals.\n\nTitle: Aditya Birla Sun Life Pure Value Fund |ABSL Pure Value Fund - ABSLMF\n\nSection: Investment Objective\n\nThe Objective of the scheme is to generate consistent long-term capital appreciation through value investing strategy.\n\nTitle: Aditya Birla Sun Life Pure Value Fund |ABSL Pure Value Fund - ABSLMF\n\nSection: Highlights\n\n### This product is suitable for investors who are seeking\\*\n* Long term capital growth\n* Investment predominantly in equity and equity-related securities following value investing strategy i.e. investing in stocks priced lower than their true value, based on fundamentals\n\n\\*Investors should consult their financial advisers if in doubt whether the product is suitable for them\n\n<img alt=\"Moderately High Risk\" src=\"https://mutualfund.adityabirlacapital.com/wealth-creation-solutions/abc5_files/ADITYA-BIRLA-SUN-LIFE-PURE-VALUE-FUND.jpg\"/>\n\n[Invest Now](https://mutualfund.adityabirlacapital.com/portal/investor/quick-invest/fund-select;fundcode=PVF?null)\n[Add to Wishlist](https://mutualfund.adityabirlacapital.com/wealth-creation-solutions/None)\n\nAdd to Compare\n\nShow More\n\nTitle: Aditya Birla Sun Life Pure Value Fund |ABSL Pure Value Fund - ABSLMF\n\nSection: Fund Details\n* Fund Type - Open-Ended\n* Category - Equity\n* Sub-Category - Value Fund\n* Min Investment - Rs. 1000/-\n* Fund Manager - Mr. Kunal Sangoi & Mr. Dhaval Joshi\n* Latest NAV - 79.914 (as on 08-Jun-2023)\n* Inception Date - Mar 28, 2008\n\nView Full Details\n\nTitle: Aditya Birla Sun Life Pure Value Fund |ABSL Pure Value Fund - ABSLMF\n\nSection: This product is suitable for investors who are seeking\n* Long term capital growth\n* Investment predominantly in equity and equity-related securities following value investing strategy i.e. investing in stocks priced lower than their true value, based on fundamentals\n\nTitle: Aditya Birla Sun Life Pure Value Fund |ABSL Pure Value Fund - ABSLMF\n\nSection: Fund Management\n\n| [Aditya Birla Sun Life Pure Value Fund](https://mutualfund.adityabirlacapital.com/wealth-creation-solutions/aditya-birla-sun-life-pure-value-fund) | 22.38 % | 27.71 % | 6.13 % | 14.64 % |\n| [Aditya Birla Sun Life Digital India Fund](https://mutualfund.adityabirlacapital.com/most-popular-funds/aditya-birla-sun-life-digital-india-fund) | 5.65 % | 31.82 % | 19.54 % | 11.27 % |\n| [Aditya Birla Sun Life Focused Equity Fund](https://mutualfund.adityabirlacapital.com/most-popular-funds/aditya-birla-sun-life-focused-equity-fund) | 13.48 % | 21.47 % | 11.01 % | 13.67 % |\n| [Aditya Birla Sun Life Digital India Fund](https://mutualfund.adityabirlacapital.com/our-best-sip-fund/aditya-birla-sun-life-digital-india-fund) | 5.65 % | 31.82 % | 19.54 % | 11.27 % |\n\nLoad More\n\nTotal Schemes managed by Mr. Kunal Sangoi is 6\n\nDifferent plans shall have a different expense structure. The performance details provided herein are of (regular / direct) plan.",
        ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)

    max_new_tokens = args.max_new_tokens
    input_ids = tokenized_inputs.input_ids.type(torch.IntTensor).to(
        'cuda')  # [batch_size, padded_length]
    # by default int64, must cast to int32! otherwise C++ kernel will interpret as [a, 0, b, 0, c, 0, ...]

    if tensorrt_llm.mpi_rank() == 0:
        print("--------------------------------------")
        print(
            f"BOS={tokenizer.bos_token_id}, PAD={tokenizer.pad_token_id}, EOS={tokenizer.eos_token_id}"
        )
        print("input text: ", input_text)
        print("input ids: ", input_ids)
        print("input lengths: ", tokenized_inputs.attention_mask.sum(dim=1))
        print("--------------------------------------")

    model_config = AutoConfig.from_pretrained(args.model_name)

    # start_id for decoder (could add more input_ids as forced_decoder_ids)
    decoder_input_ids = torch.IntTensor([[model_config.decoder_start_token_id]
                                         ]).to('cuda')
    decoder_input_ids = decoder_input_ids.repeat((input_ids.shape[0], 1))

    # simple comparison with HF on FP32
    if args.compare_hf_fp32:
        if tensorrt_llm.mpi_rank() == 0:
            if "t5" in args.model_name:
                hf_model = T5ForConditionalGeneration.from_pretrained(
                    args.model_name).to('cuda')
            else:
                pass

            tik = time.time()
            hf_output_ids = hf_model.generate(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=args.num_beams,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True)
            torch.cuda.synchronize()
            tok = time.time()

            output_ids = hf_output_ids.squeeze(dim=1)
            hf_output_text = tokenizer.batch_decode(output_ids,
                                                    skip_special_tokens=True)
            decoder_input_lengths = (decoder_input_ids !=
                                     tokenizer.pad_token_id).sum(dim=1)
            output_gen_lengths = (output_ids != tokenizer.eos_token_id).sum(
                dim=1) - decoder_input_lengths
            print("--------------------------------------")
            print("HF output_ids: ", output_ids)
            print("HF output text: ", hf_output_text)
            print("HF output generated lengths: ", output_gen_lengths)
            print(f"HF E2E time {(tok-tik)*1000}ms")
            print("--------------------------------------")
            
            ### Return logits
            hf_output_dict = hf_model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
            )
            
            yes_id = tokenizer("yes")["input_ids"][0]
            no_id = tokenizer("no")["input_ids"][0]
            
            softmax = torch.nn.Softmax(dim=1)
            logits = hf_output_dict.logits.view(-1, hf_model.config.vocab_size)
            yes_score = softmax(logits)[:, yes_id]
            no_score = softmax(logits)[:, no_id]
            score = yes_score / (yes_score + no_score)
            
            print("--------------------------------------")
            print("Inspect Logits")
            print("HF output dictionary: ", hf_output_dict.keys())
            print("HF logits: ", hf_output_dict.logits)
            print("HF logits shape: ", hf_output_dict.logits.shape)
            print("Token 'yes' id:", yes_id, " -- Token 'no' id:", no_id)
            print("Score:", score)
            print("--------------------------------------")
            
    # TRT-LLM runtime
    tllm_model = TRTLLMEncDecModel.from_engine(args.engine_name,
                                               args.engine_dir,
                                               debug_mode=args.debug_mode)
    tik = time.time()
    tllm_output_dict = tllm_model.generate(
        encoder_input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=args.num_beams,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        debug_mode=args.debug_mode,
    )
    tok = time.time()

    inference_dtype = tllm_model.encoder_model_config.dtype

    if tensorrt_llm.mpi_rank() == 0:
        
        print("--------------------------------------")
        print("TRT-LLM output keys: ", ", ".join(list(tllm_output_dict.keys())))
        print("Shape of output_ids: ", tllm_output_dict["output_ids"].shape )
        print("Shape of context_logits: ", tllm_output_dict["context_logits"].shape )
        print("context_logits: ", tllm_output_dict["context_logits"])
        print("Length of generation_logits: ", len(tllm_output_dict['generation_logits']))
        if len(tllm_output_dict['generation_logits']) > 0:
            print("Shape of one of them: ", tllm_output_dict['generation_logits'][0].shape)                       
            print("generation_logits: ", tllm_output_dict['generation_logits'][0])
        output_ids = tllm_output_dict["output_ids"][:, 0, :]
        output_text = tokenizer.batch_decode(output_ids,
                                             skip_special_tokens=True)
        decoder_input_lengths = (decoder_input_ids !=
                                 tokenizer.pad_token_id).sum(dim=1)
        output_gen_lengths = (output_ids != tokenizer.eos_token_id).sum(
            dim=1) - decoder_input_lengths
        print("--------------------------------------")
        print("TRT-LLM output_ids: ", output_ids)
        print("TRT-LLM output text: ", output_text)
        print("TRT-LLM output generated lengths: ", output_gen_lengths)
        print(f"TRT-LLM E2E time {(tok-tik)*1000}ms")
        print("Precision:", inference_dtype)
        print("--------------------------------------")

        # simple accuracy check
        if args.compare_hf_fp32:
            from difflib import SequenceMatcher
            match_rate = SequenceMatcher(None, "\n".join(output_text),
                                         "\n".join(hf_output_text)).ratio()
            print("HF output text: ", hf_output_text)
            print("HF score: ",score)
            print("TRT-LLM output text: ",output_text)
            if inference_dtype != "float32":
                print("")
                print(
                    f"[CAVEAT] Comparing TRT-LLM {inference_dtype} results with HF float32 results. Close match are not expected!"
                )
            assert match_rate > 0.9, f"Incorrect results! Match rate {match_rate}"
            print(
                f"TRT-LLM results match HF FP32 results with literal match rate {match_rate}"
            )
