from .network import Agent
from .mha_encoder_decoder import MHANodeItemEncoder, MHAItemSelectionDecoder, MHANodeSelectionDecoder


def make_network() -> Agent:
    encoder = MHANodeItemEncoder()
    idecoder = MHAItemSelectionDecoder()
    ndecoder = MHANodeSelectionDecoder()
    return Agent(encoder, idecoder, ndecoder)
