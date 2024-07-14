class ResearchDesign:
    def __init__(self, research_question, prediction_target, roi_definition, analysis_strategy):
        self.research_question = research_question
        self.prediction_target = prediction_target
        self.roi_definition = roi_definition
        self.analysis_strategy = analysis_strategy

    def display_design(self):
        print(f"Research Question: {self.research_question}")
        print(f"Prediction Target: {self.prediction_target}")
        print(f"ROI Definition: {self.roi_definition}")
        print(f"Analysis Strategy: {self.analysis_strategy}")

if __name__ == "__main__":
    design = ResearchDesign(
        research_question="Predicting disease progression using MRI data",
        prediction_target="Progression-Free Survival",
        roi_definition={"organ": "liver", "region": "tumor"},
        analysis_strategy="Use random forest for classification"
    )
    design.display_design()
