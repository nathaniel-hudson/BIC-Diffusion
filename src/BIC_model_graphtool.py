from graph_tool.all import *
from Model          import BIC_Model

class BIC_Model_gt(BIC_Model):

    def __init__(self, graph, ffm, init_opinion):
        super().__init__(graph, ffm, init_opinion)