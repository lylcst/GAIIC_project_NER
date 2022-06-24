from .modeling_nezha import NeZhaModel, NeZhaLSTMModel
from .modeling_nezha import NeZhaForTokenClassification
from ark_nlp.nn.layer.global_pointer_block import GlobalPointer, EfficientGlobalPointer
import torch


class GlobalPointerNeZha(NeZhaForTokenClassification):
    """
    GlobalPointer + Bert 的命名实体模型

    Args:
        config: 模型的配置对象
        bert_trained (:obj:`bool`, optional): 预训练模型的参数是否可训练

    Reference:
        [1] https://www.kexue.fm/archives/8373
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True,
        head_size=64
    ):
        super(GlobalPointerNeZha, self).__init__(config)

        self.num_labels = config.num_labels

        self.bert = NeZhaModel(config)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.global_pointer = GlobalPointer(
            self.num_labels,
            head_size,
            config.hidden_size
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]

        sequence_output = outputs

        logits = self.global_pointer(sequence_output, mask=attention_mask)

        return logits


class GlobalPointerLSTMNeZha(NeZhaForTokenClassification):
    """
    GlobalPointer + Bert 的命名实体模型

    Args:
        config: 模型的配置对象
        bert_trained (:obj:`bool`, optional): 预训练模型的参数是否可训练

    Reference:
        [1] https://www.kexue.fm/archives/8373
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True,
        head_size=64
    ):
        super(GlobalPointerLSTMNeZha, self).__init__(config)

        self.num_labels = config.num_labels

        self.bert = NeZhaLSTMModel(config)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.global_pointer = GlobalPointer(
            self.num_labels,
            head_size,
            config.hidden_size
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]

        sequence_output = outputs

        logits = self.global_pointer(sequence_output, mask=attention_mask)

        return logits


class GlobalPointerNeZhaFourLayer(NeZhaForTokenClassification):
    """
    GlobalPointer + Bert 的命名实体模型

    Args:
        config: 模型的配置对象
        bert_trained (:obj:`bool`, optional): 预训练模型的参数是否可训练

    Reference:
        [1] https://www.kexue.fm/archives/8373
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True,
        head_size=64
    ):
        super(GlobalPointerNeZhaFourLayer, self).__init__(config)

        self.num_labels = config.num_labels
        config.output_hidden_states = True
        config.output_attentions = False
        self.bert = NeZhaModel(config)

        self.layer_weights = [self.two_layer, self.three_layer, self.four_layer]

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.global_pointer = GlobalPointer(
            self.num_labels,
            head_size,
            config.hidden_size
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        all_hidden_states = outputs[-1]

        # 加和最后四层
        all_hidden_states = list(all_hidden_states)[-4:-1]
        for hidden_state in all_hidden_states:
            sequence_output += hidden_state
        sequence_output = sequence_output / 4.0

        logits = self.global_pointer(sequence_output, mask=attention_mask)

        return logits