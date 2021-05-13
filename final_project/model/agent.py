import os
from .resnet import ResNet
from .value import ValueNet
from .policy import PolicyNet
from utils.config import *
import logging

class Player:
    def __init__(self):
        """ Create an agent and initialize the networks """

        self.extractor = ResNet(INPLANES, OUTPLANES_MAP).to(DEVICE)
        self.value_net = ValueNet(OUTPLANES_MAP, OUTPLANES).to(DEVICE)
        self.policy_net = PolicyNet(OUTPLANES_MAP, OUTPLANES).to(DEVICE)

    def predict(self, state):
        """ Predict the probabilities and the winner from a given state """

        feature_maps = self.extractor(state)
        value = self.value_net(feature_maps)
        probs = self.policy_net(feature_maps)
        return value, probs

    def save_models(self, state, model_path):
        """ Save the models """

        for model in ["extractor", "policy_net", "value_net"]:
            self._save_checkpoint(getattr(self, model), model, state, model_path)

    def _save_checkpoint(self, model, filename, state, model_path):
        """ Save a checkpoint of the models """

        dir_path = os.path.join("./saved_models", model_path)
        if not os.path.exists(dir_path):
            logging.info("creating new directory")
            os.makedirs(dir_path)

        filename = os.path.join(dir_path, "{}_{}.pth.tar".format(state['version'], filename))
        state['model'] = model.state_dict()
        logging.info("saving")
        torch.save(state, filename)

    def load_models(self, model_path, step):
        """ Load an already saved model """

        for name in ["extractor", "policy_net", "value_net"]:
            path = model_path + "/step_{}_{}.pth.tar".format(step, name)
            checkpoint = torch.load(path)
            model = getattr(self, name)
            model.load_state_dict(checkpoint['model'])
            return checkpoint