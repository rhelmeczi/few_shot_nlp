import enum


class MultiTargetStrategy(enum.Enum):
    ONE_VS_REST = "one-vs-rest"
    MULTI_OUTPUT = "multi-output"
    CLASSIFIER_CHAIN = "classifier-chain"


class ProblemType(enum.Enum):
    SINGLE_LABEL_CLASSIFICATION = "single_label_classification"
    MULTI_LABEL_CLASSIFICATION = "multi_label_classification"


class Algorithm(enum.Enum):
    FINE_TUNING = "fine-tuning"
    SETFIT = "setfit"


class ModelingObjective(enum.Enum):
    SEQUENCE_CLASSIFICATION = "sequence-classifier"
    MASKED_LANGUAGE_MODELING = "mlm"
