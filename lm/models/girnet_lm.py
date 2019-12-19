import numpy as np

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import *
from allennlp.training.metrics import Perplexity


# class GirNetCell(torch.nn.Module):
#
#     def __init__(self, lstm1: Seq2SeqEncoder, lstm2: Seq2SeqEncoder) -> None:
#         super(GirNetCell, self).__init__()
#
#         self.lstm1 = lstm1
#         self.lstm2 = lstm2
#
#     def forward(self, input: Any, hidden: Any) -> Any:
#         lstm_1_out = self.lstm1(input)
#         lstm_2_out = self.lstm2(input)


class _SoftmaxLoss(torch.nn.Module):
    """
    Given some embeddings and some targets, applies a linear layer
    to create logits over possible words and then returns the
    negative log likelihood.
    """

    def __init__(self, num_words: int, embedding_dim: int) -> None:
        super().__init__()

        # TODO(joelgrus): implement tie_embeddings (maybe)
        self.tie_embeddings = False

        self.softmax_w = torch.nn.Parameter(
            torch.randn(embedding_dim, num_words) / np.sqrt(embedding_dim)
        )
        self.softmax_b = torch.nn.Parameter(torch.zeros(num_words))

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # embeddings is size (n, embedding_dim)
        # targets is (batch_size, ) with the correct class id
        # Does not do any count normalization / divide by batch size
        probs = torch.nn.functional.log_softmax(
            torch.matmul(embeddings, self.softmax_w) + self.softmax_b, dim=-1
        )

        return torch.nn.functional.nll_loss(probs, targets.long(), reduction="sum")


@Model.register("girnet_lm")
class GirNetLM(Model):
    """
    The ``LanguageModel`` applies a "contextualizing"
    ``Seq2SeqEncoder`` to uncontextualized embeddings, using a ``SoftmaxLoss``
    module (defined above) to compute the language modeling loss.
    should have "is_bidirectional()"

    If bidirectional is True,  the language model is trained to predict the next and
    previous tokens for each token in the input. In this case, the contextualizer must
    be bidirectional. If bidirectional is False, the language model is trained to only
    predict the next token for each token in the input; the contextualizer should also
    be unidirectional.

    If your language model is bidirectional, it is IMPORTANT that your bidirectional
    ``Seq2SeqEncoder`` contextualizer does not do any "peeking ahead". That is, for its
    forward direction it should only consider embeddings at previous timesteps, and for
    its backward direction only embeddings at subsequent timesteps. Similarly, if your
    language model is unidirectional, the unidirectional contextualizer should only
    consider embeddings at previous timesteps. If this condition is not met, your
    language model is cheating.

    Parameters
    ----------
    vocab: ``Vocabulary``
    text_field_embedder: ``TextFieldEmbedder``
        Used to embed the indexed tokens we get in ``forward``.
    contextualizer: ``Seq2SeqEncoder``
        Used to "contextualize" the embeddings. As described above,
        this encoder must not cheat by peeking ahead.
    dropout: ``float``, optional (default: None)
        If specified, dropout is applied to the contextualized embeddings before computation of
        the softmax. The contextualized embeddings themselves are returned without dropout.
    num_samples: ``int``, optional (default: None)
        If provided, the model will use ``SampledSoftmaxLoss``
        with the specified number of samples. Otherwise, it will use
        the full ``_SoftmaxLoss`` defined above.
    sparse_embeddings: ``bool``, optional (default: False)
        Passed on to ``SampledSoftmaxLoss`` if True.
    bidirectional: ``bool``, optional (default: False)
        Train a bidirectional language model, where the contextualizer
        is used to predict the next and previous token for each input token.
        This must match the bidirectionality of the contextualizer.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    aux_contextualizer : ``Seq2SeqEncoder``
    """

    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            contextualizer: Seq2SeqEncoder,
            aux_contextualizer: Seq2SeqEncoder,
            dropout: float = None,
            num_samples: int = None,
            sparse_embeddings: bool = False,
            bidirectional: bool = False,
            initializer: InitializerApplicator = None,
            regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder

        if contextualizer.is_bidirectional() is not bidirectional:
            raise ConfigurationError(
                "Bidirectionality of contextualizer must match bidirectionality of "
                "language model. "
                f"Contextualizer bidirectional: {contextualizer.is_bidirectional()}, "
                f"language model bidirectional: {bidirectional}"
            )

        self._contextualizer_lang1 = aux_contextualizer
        self._contextualizer_lang2 = copy.deepcopy(aux_contextualizer)
        self._contextualizer = contextualizer

        self._bidirectional = bidirectional
        self._bidirectional_aux = aux_contextualizer.is_bidirectional()

        # The dimension for making predictions just in the forward
        # (or backward) direction.
        # main contextualizer forward dim
        if self._bidirectional:
            self._forward_dim = contextualizer.get_output_dim() // 2
        else:
            self._forward_dim = contextualizer.get_output_dim()

        # aux contextualizer forward dim
        if self._bidirectional_aux:
            self._forward_dim_aux = aux_contextualizer.get_output_dim() // 2
        else:
            self._forward_dim_aux = aux_contextualizer.get_output_dim()

        # TODO(joelgrus): more sampled softmax configuration options, as needed.
        if num_samples is not None:
            self._lang1_softmax_loss = SampledSoftmaxLoss(
                num_words=vocab.get_vocab_size(),
                embedding_dim=self._forward_dim_aux,
                num_samples=num_samples,
                sparse=sparse_embeddings,
            )
            self._lang2_softmax_loss = SampledSoftmaxLoss(
                num_words=vocab.get_vocab_size(),
                embedding_dim=self._forward_dim_aux,
                num_samples=num_samples,
                sparse=sparse_embeddings,
            )
            self._cm_softmax_loss = SampledSoftmaxLoss(
                num_words=vocab.get_vocab_size(),
                embedding_dim=self._forward_dim,
                num_samples=num_samples,
                sparse=sparse_embeddings,
            )
        else:
            self._lang1_softmax_loss = _SoftmaxLoss(
                num_words=vocab.get_vocab_size(), embedding_dim=self._forward_dim_aux
            )
            self._lang2_softmax_loss = _SoftmaxLoss(
                num_words=vocab.get_vocab_size(), embedding_dim=self._forward_dim_aux
            )
            self._cm_loss = _SoftmaxLoss(
                num_words=vocab.get_vocab_size(), embedding_dim=self._forward_dim
            )

        # This buffer is now unused and exists only for backwards compatibility reasons.
        self.register_buffer("_last_average_loss", torch.zeros(1))

        self._lang1_perplexity = Perplexity()
        self._lang2_perplexity = Perplexity()
        self._cm_perplexity = Perplexity()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        if initializer is not None:
            initializer(self)

    # SAFE
    def _get_target_token_embeddings(
            self, token_embeddings: torch.Tensor, mask: torch.Tensor, direction: int
    ) -> torch.Tensor:
        # Need to shift the mask in the correct direction
        zero_col = token_embeddings.new_zeros(mask.size(0), 1).to(dtype=torch.bool)
        if direction == 0:
            # forward direction, get token to right
            shifted_mask = torch.cat([zero_col, mask[:, 0:-1]], dim=1)
        else:
            shifted_mask = torch.cat([mask[:, 1:], zero_col], dim=1)
        return token_embeddings.masked_select(shifted_mask.unsqueeze(-1)).view(
            -1, self._forward_dim
        )

    def is_label_bidirectional(self, label):
        if label is 'lang1' or label is 'lang2':
            return self._bidirectional_aux
        elif label is 'cm':
            return self._bidirectional
        else:
            raise Exception(f"unknown label {label}")


    def _compute_loss(
            self,
            lm_embeddings: torch.Tensor,
            token_embeddings: torch.Tensor,
            forward_targets: torch.Tensor,
            backward_targets: torch.Tensor = None,
            label="cm"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # If bidirectional, lm_embeddings is shape (batch_size, timesteps, dim * 2)
        # If unidirectional, lm_embeddings is shape (batch_size, timesteps, dim)
        # forward_targets, backward_targets (None in the unidirectional case) are
        # shape (batch_size, timesteps) masked with 0
        if self.is_label_bidirectional(label):
            forward_embeddings, backward_embeddings = lm_embeddings.chunk(2, -1)
            backward_loss = self._loss_helper(
                1, backward_embeddings, backward_targets, token_embeddings, label=label
            )
        else:
            forward_embeddings = lm_embeddings
            backward_loss = None

        forward_loss = self._loss_helper(0, forward_embeddings, forward_targets, token_embeddings, label=label)
        return forward_loss, backward_loss

    def _loss_helper(
            self,
            direction: int,
            direction_embeddings: torch.Tensor,
            direction_targets: torch.Tensor,
            token_embeddings: torch.Tensor,
            label="cm"
    ) -> Tuple[int, int]:
        mask = direction_targets > 0
        # we need to subtract 1 to undo the padding id since the softmax
        # does not include a padding dimension

        # shape (batch_size * timesteps, )
        non_masked_targets = direction_targets.masked_select(mask) - 1

        if label is "lang1" or label is "lang2":
            non_masked_embeddings = direction_embeddings.masked_select(mask.unsqueeze(-1)).view(
                -1, self._forward_dim_aux
            )
        else:
            # shape (batch_size * timesteps, embedding_dim)
            non_masked_embeddings = direction_embeddings.masked_select(mask.unsqueeze(-1)).view(
                -1, self._forward_dim
            )
        # note: need to return average loss across forward and backward
        # directions, but total sum loss across all batches.
        # Assuming batches include full sentences, forward and backward
        # directions have the same number of samples, so sum up loss
        # here then divide by 2 just below
        if label is "lang1":
            if not self._lang1_softmax_loss.tie_embeddings or not self._use_character_inputs:
                return self._lang1_softmax_loss(non_masked_embeddings, non_masked_targets)
        elif label is "lang2":
            if not self._lang2_softmax_loss.tie_embeddings or not self._use_character_inputs:
                return self._lang2_softmax_loss(non_masked_embeddings, non_masked_targets)
        elif label is "cm":
            if not self._cm_softmax_loss.tie_embeddings or not self._use_character_inputs:
                return self._cm_softmax_loss(non_masked_embeddings, non_masked_targets)

    def delete_softmax(self) -> None:
        """
        Remove the softmax weights. Useful for saving memory when calculating the loss
        is not necessary, e.g. in an embedder.
        """
        self._softmax_loss = None

    # UNSAFE
    def num_layers(self) -> int:
        """
        Returns the depth of this LM. That is, how many layers the contextualizer has plus one for
        the non-contextual layer.
        """
        if hasattr(self._contextualizer, "num_layers"):
            return self._contextualizer.num_layers + 1
        else:
            raise NotImplementedError(
                f"Contextualizer of type {type(self._contextualizer)} "
                + "does not report how many layers it has."
            )

    def forward(  # type: ignore
            self,
            lang1: Dict[str, torch.LongTensor],
            lang2: Dict[str, torch.LongTensor],
            cm: Dict[str, torch.LongTensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the averaged forward (and backward, if language model is bidirectional)
        LM loss from the batch.

        Parameters
        ----------
        lang1: ``Dict[str, torch.LongTensor]``, required.
            The output of ``Batch.as_tensor_dict()`` for a batch of sentences. By convention,
            it's required to have at least a ``"tokens"`` entry that's the output of a
            ``SingleIdTokenIndexer``, which is used to compute the language model targets.
        lang2: ``Dict[str, torch.LongTensor]``, required.
            The output of ``Batch.as_tensor_dict()`` for a batch of sentences. By convention,
            it's required to have at least a ``"tokens"`` entry that's the output of a
            ``SingleIdTokenIndexer``, which is used to compute the language model targets.
        cm: ``Dict[str, torch.LongTensor]``, required.
            The output of ``Batch.as_tensor_dict()`` for a batch of sentences. By convention,
            it's required to have at least a ``"tokens"`` entry that's the output of a
            ``SingleIdTokenIndexer``, which is used to compute the language model targets.

        Returns
        -------
        Dict with keys:

        ``'loss'``: ``torch.Tensor``
            forward negative log likelihood, or the average of forward/backward
            if language model is bidirectional
        ``'forward_loss'``: ``torch.Tensor``
            forward direction negative log likelihood
        ``'backward_loss'``: ``torch.Tensor`` or ``None``
            backward direction negative log likelihood. If language model is not
            bidirectional, this is ``None``.
        ``'lm_embeddings'``: ``Union[torch.Tensor, List[torch.Tensor]]``
            (batch_size, timesteps, embed_dim) tensor of top layer contextual representations or
            list of all layers. No dropout applied.
        ``'noncontextual_token_embeddings'``: ``torch.Tensor``
            (batch_size, timesteps, token_embed_dim) tensor of bottom layer noncontextual
            representations
        ``'mask'``: ``torch.Tensor``
            (batch_size, timesteps) mask for the embeddings

        """

        # get text field mask for each input; safe operation
        lang1_mask = get_text_field_mask(lang1)
        lang2_mask = get_text_field_mask(lang2)
        cm_mask = get_text_field_mask(cm)

        # shape (batch_size, timesteps, embedding_size)
        lang1_embeddings = self._text_field_embedder(lang1)
        lang2_embeddings = self._text_field_embedder(lang2)
        cm_embeddings = self._text_field_embedder(cm)

        # Either the top layer or all layers.
        lang1_contextual_embeddings: Union[torch.Tensor, List[torch.Tensor]] = self._contextualizer_lang1(
            lang1_embeddings, lang1_mask)
        lang2_contextual_embeddings: Union[torch.Tensor, List[torch.Tensor]] = self._contextualizer_lang2(
            lang2_embeddings, lang2_mask)

        return_dict = {}

        lang1_dict = self._each_lang_lost(
            mask=lang1_mask,
            source=lang1,
            embeddings=lang1_embeddings,
            contextual_embeddings=lang1_contextual_embeddings,
            label='lang1'
        )
        return_dict.update(lang1_dict)

        lang2_dict = self._each_lang_lost(
            mask=lang2_mask,
            source=lang2,
            embeddings=lang2_embeddings,
            contextual_embeddings=lang2_contextual_embeddings,
            label='lang2'
        )
        return_dict.update(lang2_dict)

        ## GIRNET STUFF
        # get lang1 and lang2 embedding of code_mixed data
        cm_lang1_contextual_embeddings: Union[torch.Tensor, List[torch.Tensor]] = self._contextualizer_lang1(
            cm_embeddings, cm_mask)
        cm_lang2_contextual_embeddings: Union[torch.Tensor, List[torch.Tensor]] = self._contextualizer_lang2(
            cm_embeddings, cm_mask)
        #
        # # MERGE aux representations
        # if self._bidirectional_aux:
        #     # if aux are bidirectional only consider forward part of the contextual embeddings
        #     cm_lang1_contextual_embeddings_forward, _ = cm_lang1_contextual_embeddings.chunk(2, -1)
        #     cm_lang2_contextual_embeddings_forward, _ = cm_lang2_contextual_embeddings.chunk(2, -1)
        #     cm_cat_contextual_embeddings = torch.cat(
        #         [cm_lang1_contextual_embeddings_forward, cm_lang2_contextual_embeddings_forward], -1)
        # else:
        cm_cat_contextual_embeddings = torch.cat(
            [cm_lang1_contextual_embeddings, cm_lang2_contextual_embeddings], -1)

        # Run _contextualizer on the merged representation of the input
        cm_contextual_embeddings: Union[torch.Tensor, List[torch.Tensor]] = self._contextualizer(
            cm_cat_contextual_embeddings, cm_mask)

        cm_dict = self._each_lang_lost(
            mask=cm_mask,
            source=cm,
            embeddings=cm_embeddings,
            contextual_embeddings=cm_contextual_embeddings,
            label='cm'
        )
        return_dict.update(cm_dict)

        # If we have target tokens, calculate the loss.
        token_ids = cm.get("tokens")  # safe
        if token_ids is not None:
            average_loss = (lang1_dict['lang1_loss'] + lang2_dict['lang2_loss'] + (2*cm_dict['cm_loss'])) / 4
            return_dict.update({
                "loss": average_loss
            })

        return return_dict



    def _each_lang_lost(self, mask,
                        source: Dict[str, torch.LongTensor],
                        embeddings,
                        contextual_embeddings: torch.Tensor,
                        label
                        ):

        return_dict = {}

        # If we have target tokens, calculate the loss.
        token_ids = source.get("tokens")  # safe
        if token_ids is not None:
            assert isinstance(contextual_embeddings, torch.Tensor)

            # Use token_ids to compute targets
            forward_targets = torch.zeros_like(token_ids)
            forward_targets[:, 0:-1] = token_ids[:, 1:]

            if self.is_label_bidirectional(label):
                backward_targets = torch.zeros_like(token_ids)
                backward_targets[:, 1:] = token_ids[:, 0:-1]
            else:
                backward_targets = None

            # add dropout
            contextual_embeddings_with_dropout = self._dropout(contextual_embeddings)

            # compute softmax loss
            forward_loss, backward_loss = self._compute_loss(
                contextual_embeddings_with_dropout,
                embeddings,
                forward_targets,
                backward_targets,
                label=label
            )

            num_targets = torch.sum((forward_targets > 0).long())
            if num_targets > 0:
                if self.is_label_bidirectional(label):
                    average_loss = 0.5 * (forward_loss + backward_loss) / num_targets.float()
                    # average_loss = forward_loss / num_targets.float()
                else:
                    average_loss = forward_loss / num_targets.float()
            else:
                average_loss = torch.tensor(0.0).to(forward_targets.device)

            if label is 'lang1':
                self._lang1_perplexity(average_loss)
            elif label is 'lang2':
                self._lang2_perplexity(average_loss)
            elif label is 'cm':
                self._cm_perplexity(average_loss)

            if num_targets > 0:
                return_dict.update(
                    {
                        f"{label}_loss": average_loss,
                        f"{label}_forward_loss": forward_loss / num_targets.float(),
                        f"{label}_batch_weight": num_targets.float(),
                    }
                )
                if backward_loss is not None:
                    return_dict[f"{label}_backward_loss"] = backward_loss / num_targets.float()
            else:
                # average_loss zero tensor, return it for all
                return_dict.update({f"{label}_loss": average_loss, "forward_loss": average_loss})
                if backward_loss is not None:
                    return_dict[f"{label}_backward_loss"] = average_loss

        if label is "cm":
            return_dict.update(
                {
                    "lm_embeddings": contextual_embeddings,
                    "noncontextual_token_embeddings": embeddings,
                    "mask": mask,
                }
            )
        else:
            return_dict.update(
                {
                    f"{label}_lm_embeddings": contextual_embeddings,
                    f"{label}_noncontextual_token_embeddings": embeddings,
                    f"{label}_mask": mask,
                }
            )

        return return_dict

    def get_metrics(self, reset: bool = False):
        return {
            "ppl_lang1": self._lang1_perplexity.get_metric(reset=reset),
            "ppl_lang2": self._lang2_perplexity.get_metric(reset=reset),
            "ppl_cm": self._cm_perplexity.get_metric(reset=reset)
        }
