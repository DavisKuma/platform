/**
 * Footer — Professional footer with Top Offer Academy branding
 * Design: Warm Modernism — cream bg, muted links, Sora font
 */
import { Sparkles, ExternalLink } from "lucide-react";
import { useLocation } from "wouter";

export default function Footer() {
  const [, navigate] = useLocation();

  return (
    <footer className="border-t border-border bg-card">
      <div className="container py-12">
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-8 mb-10">
          {/* Brand */}
          <div className="sm:col-span-2 lg:col-span-1">
            <div className="flex items-center gap-2.5 mb-4">
              <div className="w-8 h-8 rounded-lg bg-[oklch(0.488_0.243_264.376)] flex items-center justify-center">
                <Sparkles className="w-4 h-4 text-white" />
              </div>
              <span className="font-display font-bold text-base text-foreground">
                Micro<span className="text-[oklch(0.488_0.243_264.376)]">Sponsor</span> AI
              </span>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed max-w-xs mb-4">
              Helping international students find visa-sponsored jobs at small UK companies that actually respond.
            </p>
            <a
              href="https://topofferacademy.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-sm font-semibold text-[oklch(0.488_0.243_264.376)] hover:text-[oklch(0.441_0.243_264.376)] transition-colors"
            >
              A Top Offer Academy Product <ExternalLink className="w-3.5 h-3.5" />
            </a>
          </div>

          {/* Links */}
          <div>
            <h4 className="font-display font-semibold text-sm text-foreground mb-4">Product</h4>
            <ul className="space-y-2.5">
              <li>
                <button onClick={() => navigate("/dashboard")} className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                  Dashboard
                </button>
              </li>
              <li>
                <button onClick={() => navigate("/matches")} className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                  My Matches
                </button>
              </li>
              <li>
                <button onClick={() => navigate("/outreach")} className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                  Outreach History
                </button>
              </li>
              <li>
                <button onClick={() => navigate("/")} className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                  Browse Jobs
                </button>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="font-display font-semibold text-sm text-foreground mb-4">Resources</h4>
            <ul className="space-y-2.5">
              {["Visa Sponsor Guide", "Cold Email Tips", "Interview Prep", "Blog"].map((link) => (
                <li key={link}>
                  <a
                    href="https://topofferacademy.com"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {link}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="font-display font-semibold text-sm text-foreground mb-4">Company</h4>
            <ul className="space-y-2.5">
              <li>
                <a
                  href="https://topofferacademy.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                  Top Offer Academy
                </a>
              </li>
              {["Privacy Policy", "Terms of Service", "Contact"].map((link) => (
                <li key={link}>
                  <button className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                    {link}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-8 border-t border-border flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-muted-foreground">
            &copy; {new Date().getFullYear()} MicroSponsor AI. All rights reserved. A{" "}
            <a
              href="https://topofferacademy.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[oklch(0.488_0.243_264.376)] hover:underline font-medium"
            >
              Top Offer Academy
            </a>{" "}
            product.
          </p>
          <p className="text-xs text-muted-foreground">
            Sponsor data sourced from GOV.UK Register of Licensed Sponsors
          </p>
        </div>
      </div>
    </footer>
  );
}
