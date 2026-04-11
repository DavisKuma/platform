/**
 * Navbar — Top navigation with auth-aware links, dark mode toggle, and Custom Plan CTA
 * Design: Warm Modernism — Sora font, warm cream bg, indigo accents
 */
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Sparkles, Menu, X, Zap, Moon, Sun, LogIn, LogOut, User, Shield } from "lucide-react";
import { useAuth } from "@/_core/hooks/useAuth";
import { useTheme } from "@/contexts/ThemeContext";
import { useLocation } from "wouter";

interface NavbarProps {
  activeSection?: string;
  onNavigate?: (section: string) => void;
  onOpenPricing?: () => void;
}

export default function Navbar({ activeSection, onNavigate, onOpenPricing }: NavbarProps) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const { user, isAuthenticated, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const [location, navigate] = useLocation();

  const handleNav = (path: string) => {
    navigate(path);
    setMobileOpen(false);
  };

  const isActive = (path: string) => {
    if (activeSection) {
      const sectionMap: Record<string, string> = {
        dashboard: "/dashboard",
        matches: "/matches",
        outreach: "/outreach",
      };
      return sectionMap[activeSection] === path;
    }
    return location === path;
  };

  const navItems = isAuthenticated
    ? [
        { path: "/dashboard", label: "Dashboard" },
        { path: "/matches", label: "My Matches" },
        { path: "/outreach", label: "Outreach" },
        ...(user?.role === "admin" ? [{ path: "/admin", label: "Admin" }] : []),
      ]
    : [];

  return (
    <nav className="sticky top-0 z-50 bg-background/85 backdrop-blur-xl border-b border-border/50">
      <div className="container flex items-center justify-between h-16">
        {/* Logo */}
        <button
          onClick={() => handleNav("/")}
          className="flex items-center gap-2.5 group"
        >
          <div className="w-9 h-9 rounded-xl bg-[oklch(0.488_0.243_264.376)] flex items-center justify-center shadow-md shadow-[oklch(0.488_0.243_264.376/0.25)] group-hover:shadow-lg group-hover:shadow-[oklch(0.488_0.243_264.376/0.35)] transition-shadow duration-300">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <span className="font-display font-bold text-lg tracking-tight text-foreground">
            Micro<span className="text-[oklch(0.488_0.243_264.376)]">Sponsor</span> AI
          </span>
        </button>

        {/* Desktop Nav */}
        <div className="hidden md:flex items-center gap-1">
          {navItems.map((item) => (
            <button
              key={item.path}
              onClick={() => handleNav(item.path)}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
                isActive(item.path)
                  ? "bg-[oklch(0.488_0.243_264.376/0.1)] text-[oklch(0.488_0.243_264.376)] font-semibold"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent"
              }`}
            >
              {item.path === "/admin" && <Shield className="w-3.5 h-3.5 inline mr-1" />}
              {item.label}
            </button>
          ))}
        </div>

        {/* Right side: Dark mode + Auth + CTA */}
        <div className="flex items-center gap-2">
          {/* Dark mode toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-xl hover:bg-accent transition-colors text-muted-foreground hover:text-foreground"
            aria-label="Toggle dark mode"
          >
            {theme === "dark" ? <Sun className="w-4.5 h-4.5" /> : <Moon className="w-4.5 h-4.5" />}
          </button>

          {/* Auth */}
          {isAuthenticated ? (
            <div className="hidden sm:flex items-center gap-2">
              <span className="text-xs text-muted-foreground truncate max-w-[120px]">
                {user?.name ?? user?.email ?? ""}
              </span>
              <button
                onClick={() => logout()}
                className="p-2 rounded-xl hover:bg-accent transition-colors text-muted-foreground hover:text-foreground"
                title="Sign out"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <Button
              variant="ghost"
              size="sm"
              className="hidden sm:flex rounded-xl text-sm"
              onClick={() => navigate("/")}
            >
              <LogIn className="w-4 h-4 mr-1" /> Sign In
            </Button>
          )}

          {/* Custom Plan CTA */}
          <Button
            onClick={() => onOpenPricing?.()}
            className="hidden sm:flex items-center gap-2 bg-[oklch(0.488_0.243_264.376)] hover:bg-[oklch(0.441_0.243_264.376)] text-white rounded-xl font-display font-semibold text-sm shadow-md shadow-[oklch(0.488_0.243_264.376/0.25)] hover:shadow-lg hover:shadow-[oklch(0.488_0.243_264.376/0.35)] transition-all duration-300"
          >
            <Zap className="w-4 h-4" />
            Custom Plan
          </Button>

          {/* Mobile Toggle */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="md:hidden p-2 rounded-xl hover:bg-accent transition-colors"
          >
            {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      {mobileOpen && (
        <div className="md:hidden border-t border-border/50 bg-background/95 backdrop-blur-xl">
          <div className="container py-4 space-y-1">
            {!isAuthenticated && (
              <button
                onClick={() => handleNav("/")}
                className="w-full text-left px-4 py-3 rounded-xl text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-accent flex items-center gap-2"
              >
                <LogIn className="w-4 h-4" /> Sign In
              </button>
            )}
            {navItems.map((item) => (
              <button
                key={item.path}
                onClick={() => handleNav(item.path)}
                className={`w-full text-left px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 ${
                  isActive(item.path)
                    ? "bg-[oklch(0.488_0.243_264.376/0.1)] text-[oklch(0.488_0.243_264.376)] font-semibold"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent"
                }`}
              >
                {item.label}
              </button>
            ))}
            {isAuthenticated && (
              <button
                onClick={() => { logout(); setMobileOpen(false); }}
                className="w-full text-left px-4 py-3 rounded-xl text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-accent flex items-center gap-2"
              >
                <LogOut className="w-4 h-4" /> Sign Out
              </button>
            )}
            <Button
              onClick={() => { onOpenPricing?.(); setMobileOpen(false); }}
              className="w-full mt-3 bg-[oklch(0.488_0.243_264.376)] hover:bg-[oklch(0.441_0.243_264.376)] text-white rounded-xl font-display font-semibold text-sm"
            >
              <Zap className="w-4 h-4 mr-2" />
              Custom Plan
            </Button>
          </div>
        </div>
      )}
    </nav>
  );
}
