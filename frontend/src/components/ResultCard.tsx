import type { AnalysisResult } from "../types/AnalysisResult";
import { Card } from "./Card";

interface Props {
  data: AnalysisResult;
}

export default function ResultCard({ data }: Props) {
  const diagnosisColor =
    data.results.diagnosis.prediction === "HEALTHY" ? "#22c55e" : "#ef4444";

  return (
    <Card title="Analysis Results" className="fade-in">
      <div style={{ marginBottom: "1.5rem" }}>
        <h3
          style={{
            fontSize: "0.875rem",
            textTransform: "uppercase",
            letterSpacing: "1px",
            color: "var(--text-secondary)",
            marginBottom: "0.5rem",
          }}
        >
          Diagnosis
        </h3>
        <p
          style={{
            fontSize: "1.5rem",
            fontWeight: 700,
            color: diagnosisColor,
          }}
        >
          {data.results.diagnosis.prediction}
        </p>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "1rem",
          backgroundColor: "var(--bg-color)",
          padding: "1rem",
          borderRadius: "var(--radius-md)",
          marginBottom: "1rem",
        }}
      >
        <div>
          <h4 style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>
            Predicted MMSE
          </h4>
          <p style={{ fontSize: "1.25rem", fontWeight: 600 }}>
            {data.results.mmse.predicted.toFixed(2)}
          </p>
        </div>
        <div>
          <h4 style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>
            Cognitive Level
          </h4>
          <p style={{ fontSize: "1.25rem", fontWeight: 600 }}>
            {data.results.mmse.cognitive_level}
          </p>
        </div>
      </div>

      {data.results.mmse.error_note && (
        <p
          style={{
            fontSize: "0.875rem",
            color: "#eab308",
            marginBottom: "1rem",
            padding: "0.5rem",
            backgroundColor: "rgba(234, 179, 8, 0.1)",
            borderRadius: "var(--radius-sm)",
          }}
        >
          ⚠️ {data.results.mmse.error_note}
        </p>
      )}

      <hr
        style={{
          border: "0",
          borderTop: "1px solid var(--border-color)",
          margin: "1rem 0",
        }}
      />

      <small
        style={{
          color: "var(--text-muted)",
          display: "block",
          fontStyle: "italic",
        }}
      >
        {data.disclaimer}
      </small>
    </Card>
  );
}
