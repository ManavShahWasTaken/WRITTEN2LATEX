import torch
from torch.nn import functional as F
from build_vocab import END_TOKEN, PAD_TOKEN, START_TOKEN
from .beam_search import BeamSearch

class LatexProducerUpdated(object):
    """
    Model wrapper, implementing batch greedy decoding and
    batch beam search decoding
    """

    def __init__(self, model, vocab, beam_size=5, max_len=150, use_cuda=True, model_type='transformer'):
        """args:
            the path to model checkpoint
        """
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = model.to(self.device)
        self.model_type = model_type
        self._sign2id = vocab.sign2id
        self._id2sign = vocab.id2sign
        self.max_len = max_len
        self.beam_size = beam_size
        self._beam_search = BeamSearch(END_TOKEN, max_len, beam_size)

    def __call__(self, imgs):
        """args:
            imgs: images need to be decoded
            beam_size: if equal to 1, use greedy decoding
           returns:
            formulas list of batch_size length
        """
        if self.beam_size == 1:
            if self.model_type == "transformer":
                result = self._greedy_decoding_transformer(imgs)
            else:
                result = self._batch_beam_search_LSTM(imgs)
        else:
            if self.model_type == "transformer":
                results = self._batch_beam_search_transformer(imgs)
            else:
                results = self._batch_beam_search_LSTM(imgs)
        
        return results

    def _greedy_decoding_LSTM(self, imgs):
        imgs = imgs.to(self.device)
        self.model.eval()

        enc_outs = self.model.encode(imgs)
        dec_states, O_t = self.model.init_decoder(enc_outs)

        batch_size = imgs.size(0)
        # storing decoding results
        formulas_idx = torch.ones(
            batch_size, self.max_len, device=self.device).long() * PAD_TOKEN
        # first decoding step's input
        tgt = torch.ones(
            batch_size, 1, device=self.device).long() * START_TOKEN
        with torch.no_grad():
            for t in range(self.max_len):
                dec_states, O_t, logit = self.model.step_decoding(
                    dec_states, O_t, enc_outs, tgt)

                tgt = torch.argmax(logit, dim=1, keepdim=True)
                formulas_idx[:, t:t + 1] = tgt
        results = self._idx2formulas(formulas_idx)
        return results

    def _greedy_decoding_transformer(self, imgs):
        imgs = imgs.to(self.device)
        self.model.eval()

        memory = self.model(imgs)

        batch_size = imgs.size(0)
        # storing decoding results
        formulas_idx = torch.ones(
            batch_size, self.max_len, device=self.device).long() * PAD_TOKEN
        
        # first decoding step's input
        formulas_idx[:, 0] = START_TOKEN
        # tgt = torch.ones(
        #     batch_size, 1, device=self.device).long() * START_TOKEN
        with torch.no_grad():
            for t in range(1, self.max_len):
                y = formulas_idx[:, :t]
                batch_logits = self.model.decode(y, memory)

                tgt = torch.argmax(batch_logits, dim=1, keepdim=True)
                formulas_idx[:, t:t + 1] = tgt
        results = self._idx2formulas(formulas_idx)
        return results

    def _simple_beam_search_decoding(self, imgs):
        """simple beam search decoding (not support batch)"""
        self.model.eval()
        beam_results = [
            self._bs_decoding_LSTM(img.unsqueeze(0))
            for img in imgs
        ]
        return beam_results

    def _idx2formulas(self, formulas_idx):
        """convert formula id matrix to formulas list"""
        results = []
        for id_ in formulas_idx:
            id_list = id_.tolist()
            result = []
            for sign_id in id_list:
                if sign_id != END_TOKEN:
                    result.append(self._id2sign[sign_id])
                else:
                    break
            results.append(" ".join(result))
        return results

    def _bs_decoding_LSTM(self, img):
        """
        beam search decoding not support batch
        args:
            img: [1, C, H, W]
            beam_size: int
        return:
            formulas in str format
        """
        self.model.eval()
        img = img.to(self.device)

        # encoding
        # img = img.unsqueeze(0)  # [1, C, H, W]
        enc_outs = self.model.encode(img)  # [1, H*W, OUT_C]

        # prepare data for decoding
        enc_outs = enc_outs.expand(self.beam_size, -1, -1)
        # [Beam_size, dec_rnn_h]
        dec_states, O_t = self.model.init_decoder(enc_outs)

        # store top k ids (k is less or equal to beam_size)
        # in first decoding step, all they are  start token
        topk_ids = torch.ones(
            self.beam_size, device=self.device).long() * START_TOKEN
        topk_log_probs = torch.Tensor([0.0] + [-1e10] * (self.beam_size - 1))
        topk_log_probs = topk_log_probs.to(self.device)
        seqs = torch.ones(
            self.beam_size, 1, device=self.device).long() * START_TOKEN
        # store complete sequences and corrosponing scores
        complete_seqs = []
        complete_seqs_scores = []
        k = self.beam_size
        vocab_size = len(self._sign2id)
        with torch.no_grad():
            for t in range(self.max_len):
                dec_states, O_t, logit = self.model.step_decoding(
                    dec_states, O_t, enc_outs, topk_ids.unsqueeze(1))
                log_probs = torch.log(logit)  # [k, vocab_size]

                log_probs += topk_log_probs.unsqueeze(1)
                topk_log_probs, topk_ids = torch.topk(log_probs.view(-1), k)

                beam_index = topk_ids // vocab_size
                topk_ids = topk_ids % vocab_size

                seqs = torch.cat(
                    [seqs.index_select(0, beam_index), topk_ids.unsqueeze(1)],
                    dim=1
                )

                complete_inds = [
                    ind for ind, next_word in enumerate(topk_ids)
                    if next_word == END_TOKEN
                ]
                if t == (self.max_len-1):  # last_step, end all seqs
                    complete_inds = list(range(len(topk_ids)))

                incomplete_inds = list(
                    set(range(len(topk_ids))) - set(complete_inds)
                )
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds])
                    complete_seqs_scores.extend(topk_log_probs[complete_inds])
                k -= len(complete_inds)
                if k == 0:  # all beam finished
                    break

                # prepare for next step
                seqs = seqs[incomplete_inds]
                topk_ids = topk_ids[incomplete_inds]
                topk_log_probs = topk_log_probs[incomplete_inds]

                enc_outs = enc_outs[:k]
                seleted = beam_index[incomplete_inds]
                O_t = O_t[seleted]
                dec_states = (dec_states[0][seleted],
                              dec_states[1][seleted])

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i][1:]
        result = self._idx2formulas(seq.unsqueeze(0))[0]
        return result

    def _batch_beam_search_transformer(self, imgs):
        self.model.eval()
        imgs = imgs.to(self.device)
        memory = self.model.encode(imgs)  # [batch_size, H*W, OUT_C]
        batch_size = imgs.size(0)
        start_predictions = torch.ones(
            batch_size, device=self.device).long() * START_TOKEN
        
        state = {}
        state['memory'] = memory
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self._take_step_transformer)
        
        all_top_predictions = all_top_k_predictions[:, 0, :]
        all_top_predictions = self._idx2formulas(all_top_predictions)
        return all_top_predictions

        


    def _batch_beam_search_LSTM(self, imgs):
        self.model.eval()
        imgs = imgs.to(self.device)
        enc_outs = self.model.encode(imgs)  # [batch_size, H*W, OUT_C]
        # enc_outs = enc_outs.expand(self.beam_size, -1, -1)
        dec_states, O_t = self.model.init_decoder(enc_outs)

        batch_size = imgs.size(0)
        start_predictions = torch.ones(
            batch_size, device=self.device).long() * START_TOKEN
        state = {}
        state['h_t'] = dec_states[0]
        state['c_t'] = dec_states[1]
        state['o_t'] = O_t
        state['enc_outs'] = enc_outs
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self._take_step_LSTM)

        all_top_predictions = all_top_k_predictions[:, 0, :]
        all_top_predictions = self._idx2formulas(all_top_predictions)
        return all_top_predictions

    def _take_step_LSTM(self, last_predictions, state):
        dec_states = (state['h_t'], state['c_t'])
        O_t = state['o_t']
        enc_outs = state['enc_outs']

        last_predictions = last_predictions.unsqueeze(1)
        with torch.no_grad():
            dec_states, O_t, logit = self.model.step_decoding(
                dec_states, O_t, enc_outs, last_predictions)

        # update state
        state['h_t'] = dec_states[0]
        state['c_t'] = dec_states[1]
        state['o_t'] = O_t
        return (torch.log(logit), state)
    
    def _take_step_transformer(self, last_predictions, state):
        memory = state['memory']

        last_predictions = last_predictions.unsqueeze(1)
        with torch.no_grad():
            logits = self.model.decode(last_predictions, memory)
            logits = torch.squeeze(F.log_softmax(logits, dim=-1), dim=1)

        return (logits, state)


