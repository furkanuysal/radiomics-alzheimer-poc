import React from "react";

interface CardProps {
  children: React.ReactNode;
  title?: string;
  className?: string;
}

export const Card: React.FC<CardProps> = ({
  children,
  title,
  className = "",
}) => {
  return (
    <div
      className={className}
      style={{
        backgroundColor: "var(--surface-color)",
        borderRadius: "var(--radius-lg)",
        padding: "1.5rem",
        boxShadow: "var(--shadow-md)",
        border: "1px solid var(--border-color)",
        marginBottom: "1.5rem",
      }}
    >
      {title && (
        <h2
          style={{
            fontSize: "1.25rem",
            marginBottom: "1rem",
            paddingBottom: "0.75rem",
            borderBottom: "1px solid var(--border-color)",
          }}
        >
          {title}
        </h2>
      )}
      {children}
    </div>
  );
};
