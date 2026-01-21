import React, { useState } from "react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md" | "lg";
  fullWidth?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = "primary",
  size = "md",
  fullWidth = false,
  className = "",
  style,
  ...props
}) => {
  const [isHovered, setIsHovered] = useState(false);

  const baseStyles: React.CSSProperties = {
    padding:
      size === "sm"
        ? "0.5rem 1rem"
        : size === "lg"
          ? "1rem 2rem"
          : "0.75rem 1.5rem",
    fontSize: size === "sm" ? "0.875rem" : "1rem",
    borderRadius: "var(--radius-md)",
    fontWeight: 600,
    border: "none",
    transition: "all 0.2s ease",
    width: fullWidth ? "100%" : "auto",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "0.5rem",
    opacity: props.disabled ? 0.6 : 1,
    cursor: props.disabled ? "not-allowed" : "pointer",
    ...style,
  };

  const variantStyles = {
    primary: {
      backgroundColor: "var(--primary-color)",
      color: "white",
      boxShadow: "var(--shadow-md)",
    },
    secondary: {
      backgroundColor: "var(--surface-color)",
      color: "var(--text-primary)",
      border: "1px solid var(--border-color)",
    },
    ghost: {
      backgroundColor: "transparent",
      color: "var(--text-secondary)",
    },
    danger: {
      backgroundColor: "#ef4444",
      color: "white",
    },
  };

  const hoverStyles = {
    primary: { backgroundColor: "var(--primary-hover)" },
    secondary: {
      backgroundColor: "var(--surface-hover)",
      borderColor: "var(--text-secondary)",
    },
    ghost: {
      backgroundColor: "var(--surface-hover)",
      color: "var(--text-primary)",
    },
    danger: { backgroundColor: "#dc2626" },
  };

  const currentStyles = {
    ...baseStyles,
    ...variantStyles[variant],
    ...(isHovered && !props.disabled ? hoverStyles[variant] : {}),
  };

  return (
    <button
      className={className}
      style={currentStyles}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      {...props}
    >
      {children}
    </button>
  );
};
