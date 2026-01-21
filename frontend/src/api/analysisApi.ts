import type { AnalysisResult } from "../types/AnalysisResult";

export async function analyzeMRI(
  imgFile: File,
  hdrFile: File,
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("img_file", imgFile);
  formData.append("hdr_file", hdrFile);

  const res = await fetch("http://127.0.0.1:8000/analyze-mri", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Analysis failed");
  }

  return res.json();
}
