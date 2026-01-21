export interface AnalysisResult {
  status: "ok" | "error";
  results: {
    diagnosis: {
      prediction: "HEALTHY" | "ALZHEIMER";
    };
    mmse: {
      predicted: number;
      cognitive_level: string;
      error_note: string;
    };
  };
  disclaimer: string;
}
