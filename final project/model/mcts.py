class Node:
    def __init__(self, parent=None, prob=None, move=None):
        """
            p : probability of reaching that node, given by the policy net
            n : number of time this node has been visited during simulations
            w : total action value, given by the value network
            q : mean action value (w / n)
        """
        self.p = prob
        self.n = 0
        self.w = 0
        self.q = 0
        self.children = []
        self.parent = parent
        self.move = move

    def update(self, v):
        """ Update the node statistics after a playout """

        self.w = self.w + v
        self.q = self.w / self.n if self.n > 0 else 0

    def is_leaf(self):
        """ Check whether node is a leaf or not """

        return len(self.childrens) == 0

    def expand(self, probas):
        """ Create a child node for every non-zero move probability """

        self.childrens = [Node(parent=self, move=idx, proba=probas[idx]) \
                          for idx in range(probas.shape[0]) if probas[idx] > 0]
