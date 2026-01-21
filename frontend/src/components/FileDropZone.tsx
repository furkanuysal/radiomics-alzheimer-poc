import React, { useRef, useState } from "react";
import { Button } from "./Button";

interface FileDropZoneProps {
  label: string;
  accept: string;
  onSelect: (file: File) => void;
  selectedFile: File | null;
}

export const FileDropZone: React.FC<FileDropZoneProps> = ({
  label,
  accept,
  onSelect,
  selectedFile,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files?.[0]) {
      onSelect(e.dataTransfer.files[0]);
    }
  };

  const handleClick = () => {
    inputRef.current?.click();
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
      style={{
        border: `2px dashed ${isDragOver ? "var(--primary-color)" : "var(--border-color)"}`,
        borderRadius: "var(--radius-lg)",
        padding: "2rem",
        textAlign: "center",
        cursor: "pointer",
        backgroundColor: isDragOver ? "var(--surface-hover)" : "transparent",
        transition: "all 0.2s ease",
        marginBottom: "1rem",
        position: "relative",
      }}
    >
      <input
        type="file"
        ref={inputRef}
        accept={accept}
        onChange={(e) => e.target.files?.[0] && onSelect(e.target.files[0])}
        style={{ display: "none" }}
      />

      <div style={{ pointerEvents: "none" }}>
        <div
          style={{
            fontSize: "2rem",
            marginBottom: "1rem",
            color: isDragOver
              ? "var(--primary-color)"
              : "var(--text-secondary)",
          }}
        >
          {selectedFile ? "📄" : "📁"}
        </div>

        <h3
          title={selectedFile ? selectedFile.name : label}
          style={{
            fontSize: "1rem",
            fontWeight: 500,
            marginBottom: "0.5rem",
            color: "var(--text-primary)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            maxWidth: "100%",
            padding: "0 1rem",
          }}
        >
          {selectedFile ? selectedFile.name : label}
        </h3>

        {!selectedFile && (
          <p
            style={{
              fontSize: "0.875rem",
              color: "var(--text-secondary)",
            }}
          >
            Drag & drop or click to select
          </p>
        )}

        {selectedFile && (
          <Button
            size="sm"
            variant="secondary"
            onClick={(e) => {
              e.stopPropagation();
              inputRef.current?.click();
            }}
            style={{ marginTop: "1rem", pointerEvents: "auto" }}
          >
            Change File
          </Button>
        )}
      </div>
    </div>
  );
};
