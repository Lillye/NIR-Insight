class Pipeline:

    def __init__(self):
        self.pipeline = []

    def Add(self, pipelineElement):
        self.pipeline.append(pipelineElement)
    
    def Run(self, input):
        stageOutputs = []
        for i in range(len(self.pipeline)):
            input = self.pipeline[i].Process(input, i);
            stageOutputs.append(input)
        return stageOutputs
            