import os
from model.resnet import ResNet
from model.value import ValueNet
from model.policy import PolicyNet
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

    def save_models(self, state, round, best_player=False):
        """ Save the models """

        for model in ["extractor", "policy_net", "value_net"]:
            self._save_checkpoint(getattr(self, model), model, state, round, best_player)

    def _save_checkpoint(self, model, filename, state, round, best_player=False):
        """ Save a checkpoint of the models """

        if best_player:
            dir_path = "./best_player"
        else:
            dir_path = "./saved_models/{}".format(round)
        if not os.path.exists(dir_path):
            logging.info("creating new directory")
            os.makedirs(dir_path)

        filename = os.path.join(dir_path, "{}.pth.tar".format(filename))
        state['model'] = model.state_dict()
        torch.save(state, filename)

    def load_models(self, model_path):
        """ Load an already saved model """

        for name in ["extractor", "policy_net", "value_net"]:
            path = model_path + "/{}.pth.tar".format(name)
            checkpoint = torch.load(path)
            model = getattr(self, name)
            model.load_state_dict(checkpoint['model'])
            return checkpoint