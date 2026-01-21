import { useState } from "react";
import { Layout } from "../components/Layout";
import { Card } from "../components/Card";
import { FileDropZone } from "../components/FileDropZone";
import { Button } from "../components/Button";
import { Loader } from "../components/Loader";
import ResultCard from "../components/ResultCard";
import type { AnalysisResult } from "../types/AnalysisResult";
import { analyzeMRI } from "../api/analysisApi";

export default function Home() {
  const [imgFile, setImgFile] = useState<File | null>(null);
  const [hdrFile, setHdrFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!imgFile || !hdrFile) return;

    setLoading(true);
    setResult(null);

    try {
      const data = await analyzeMRI(imgFile, hdrFile);
      setResult(data);
    } catch (e) {
      alert("Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout>
      <div style={{ maxWidth: 800, margin: "0 auto" }}>
        <div style={{ textAlign: "center", marginBottom: "3rem" }}>
          <h1
            style={{
              fontSize: "2.5rem",
              marginBottom: "1rem",
              background:
                "linear-gradient(to right, var(--primary-color), var(--accent-color))",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              display: "inline-block",
            }}
          >
            Radiomics-Based Alzheimer's Analysis
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "1.125rem" }}>
            Upload MRI scan files (.img and .hdr) for cognitive analysis.
          </p>
        </div>

        {!result && !loading && (
          <Card>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
                gap: "2rem",
              }}
            >
              <FileDropZone
                label="Upload Image File (.img)"
                accept=".img"
                onSelect={setImgFile}
                selectedFile={imgFile}
              />

              <FileDropZone
                label="Upload Header File (.hdr)"
                accept=".hdr"
                onSelect={setHdrFile}
                selectedFile={hdrFile}
              />
            </div>

            <div
              style={{
                marginTop: "2rem",
                display: "flex",
                justifyContent: "center",
              }}
            >
              <Button
                size="lg"
                disabled={!imgFile || !hdrFile}
                onClick={handleAnalyze}
                fullWidth={false}
                style={{ minWidth: "200px" }}
              >
                Analyze Scan
              </Button>
            </div>
          </Card>
        )}

        {loading && (
          <Card>
            <Loader text="Analyzing MRI scan. Please wait..." />
          </Card>
        )}

        {result && (
          <div className="fade-in">
            <ResultCard data={result} />
            <div style={{ marginTop: "2rem", textAlign: "center" }}>
              <Button
                variant="secondary"
                onClick={() => {
                  setResult(null);
                  setImgFile(null);
                  setHdrFile(null);
                }}
              >
                Analyze New Scan
              </Button>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
