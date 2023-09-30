from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    squeezenet1_1,
)
import torch.nn as nn
import torch
from base.models.model_container import ModelContainer


class CNNContainer(ModelContainer):
    def __init__(self, cfg, **kwargs):
        super(CNNContainer, self).__init__(cfg)
        print(f"Initializing model container {self.__class__.__name__}...")
        self.gen_model()

    def gen_model(self):
        """
        Generate models for self.models
        """
        use_pretrained = getattr(self.cfg, "pretrained", False)
        num_classes = getattr(self.cfg, "num_classes", 1000)
        pretrained_num_classes = getattr(self.cfg, "pretrained_num_classes", 1000)

        # Choose between models
        if self.cfg.model == "ResNet18":
            model = resnet18(
                pretrained=use_pretrained, num_classes=pretrained_num_classes
            )
        elif self.cfg.model == "ResNet34":
            model = resnet34(
                pretrained=use_pretrained, num_classes=pretrained_num_classes
            )
        elif self.cfg.model == "ResNet50":
            model = resnet50(
                pretrained=use_pretrained, num_classes=pretrained_num_classes
            )
        elif self.cfg.model == "ResNet101":
            model = resnet101(
                pretrained=use_pretrained, num_classes=pretrained_num_classes
            )
        elif self.cfg.model == "ResNet152":
            model = resnet152(
                pretrained=use_pretrained, num_classes=pretrained_num_classes
            )
        elif self.cfg.model == "SqueezeNet1_1":
            model = squeezenet1_1(
                pretrained=use_pretrained, num_classes=pretrained_num_classes
            )
        else:
            raise AttributeError("Invalid model name")

        if num_classes != pretrained_num_classes:
            model.fc = nn.Linear(512, num_classes)

        # Get channel size
        channels = getattr(self.cfg, "channel_size", 4)
        kernel_size = getattr(self.cfg, "kernel_size", 7)
        # Adapt model for 4 channels
        if "ResNet" in self.cfg.model:
            model.conv1 = nn.Conv2d(
                channels, 64, kernel_size=kernel_size, stride=2, padding=3, bias=False
            )
        elif "SqueezeNet1_1" == self.cfg.model:
            model.features[0] = nn.Conv2d(4, 64, kernel_size=7, stride=2)

        freeze_weights = getattr(self.cfg, "freeze", False)

        if freeze_weights:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.conv1.parameters():
                param.requires_grad = True
            for param in model.layer1.parameters():
                param.requires_grad = True

        train_classifier = getattr(self.cfg, "train_classifier", False)

        if train_classifier:
            print("Training only last layer!")
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True

        print(f"Using {self.cfg.model} for training")

        self.models["model"] = model

    def load_saved(self):
        # Load if pretrained
        pretrained_num_classes = getattr(self.cfg, "pretrained_num_classes", None)

        if pretrained_num_classes is not None:
            self.models["model"].fc = nn.Linear(512, pretrained_num_classes)

        if self.cfg.load_model is not None:
            print(f"Loading model from {self.cfg.load_model}")
            state_dict = torch.load(self.cfg.load_model)
            self.models["model"].load_state_dict(state_dict["state_dict"])

        if self.cfg.mode != "test":
            num_classes = getattr(self.cfg, "num_classes", 1000)
            keep_fc = getattr(self.cfg, "keep_fc", True)

            if not keep_fc:
                self.models["model"].fc = nn.Linear(512, num_classes)
