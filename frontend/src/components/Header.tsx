import React from "react";

export const Header: React.FC = () => {
  return (
    <header
      style={{
        padding: "1.5rem 2rem",
        backgroundColor: "var(--surface-color)",
        borderBottom: "1px solid var(--border-color)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        position: "sticky",
        top: 0,
        zIndex: 10,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
        <h1 style={{ fontSize: "1.25rem", margin: 0 }}>
          Radiomics Alzheimer POC
        </h1>
      </div>
      <nav>{/* Placeholder for future nav items */}</nav>
    </header>
  );
};
