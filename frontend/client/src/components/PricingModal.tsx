/**
 * PricingModal — Custom Plan pricing display
 * Design: Warm Modernism — warm card, indigo CTA, emerald checkmarks
 */
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Check, Zap, Crown, TrendingUp, Mail, Users, BarChart3, Star } from "lucide-react";
import { motion } from "framer-motion";
import { toast } from "sonner";

interface PricingModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function PricingModal({ isOpen, onClose }: PricingModalProps) {
  const [billingCycle, setBillingCycle] = useState<"monthly" | "yearly">("monthly");

  const handleActivate = () => {
    toast.info("Payments coming soon! Stay tuned.");
  };

  const features = [
    { icon: Users, label: "Unlimited hiring manager contacts", description: "Verified emails for every matched company" },
    { icon: Mail, label: "Unlimited AI-generated emails", description: "Personalised cold outreach at scale" },
    { icon: TrendingUp, label: "Priority match algorithm", description: "Get matched before free-tier users" },
    { icon: BarChart3, label: "Outreach analytics dashboard", description: "Track opens, replies, and conversion rates" },
    { icon: Crown, label: "Early access to new features", description: "Beta features and sponsor data updates" },
    { icon: Star, label: "Dedicated support", description: "Priority email support within 24 hours" },
  ];

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-lg max-h-[90vh] overflow-y-auto bg-card border-border rounded-2xl p-0">
        <div className="p-6 sm:p-8">
          <DialogHeader className="mb-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-11 h-11 rounded-xl bg-[oklch(0.488_0.243_264.376)] flex items-center justify-center shadow-md shadow-[oklch(0.488_0.243_264.376/0.3)]">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div>
                <DialogTitle className="font-display font-bold text-xl text-foreground">
                  Custom Plan
                </DialogTitle>
                <p className="text-sm text-muted-foreground">Unlock the full power of MicroSponsor AI</p>
              </div>
            </div>
          </DialogHeader>

          {/* Billing toggle */}
          <div className="flex items-center justify-center gap-1 p-1 bg-muted rounded-xl mb-8">
            <button
              onClick={() => setBillingCycle("monthly")}
              className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-semibold transition-all duration-200 ${
                billingCycle === "monthly"
                  ? "bg-card text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Monthly
            </button>
            <button
              onClick={() => setBillingCycle("yearly")}
              className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-semibold transition-all duration-200 relative ${
                billingCycle === "yearly"
                  ? "bg-card text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Yearly
              <span className="absolute -top-2 -right-1 px-1.5 py-0.5 bg-[oklch(0.696_0.17_162.48)] text-white text-[10px] font-bold rounded-full">
                -36%
              </span>
            </button>
          </div>

          {/* Price */}
          <motion.div
            key={billingCycle}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="text-center mb-8"
          >
            <div className="flex items-baseline justify-center gap-1">
              <span className="font-display font-extrabold text-5xl text-foreground">
                £{billingCycle === "monthly" ? "29" : "199"}
              </span>
              <span className="text-lg text-muted-foreground font-medium">
                /{billingCycle === "monthly" ? "mo" : "year"}
              </span>
            </div>
            {billingCycle === "yearly" && (
              <p className="text-sm text-[oklch(0.596_0.145_163.225)] font-medium mt-2">
                That's just £16.58/mo — save £149/year
              </p>
            )}
          </motion.div>

          {/* Features */}
          <div className="space-y-4 mb-8">
            {features.map((feature, i) => (
              <motion.div
                key={feature.label}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: i * 0.06 }}
                className="flex items-start gap-3"
              >
                <div className="w-8 h-8 rounded-lg bg-[oklch(0.696_0.17_162.48/0.1)] flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Check className="w-4 h-4 text-[oklch(0.596_0.145_163.225)]" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-foreground">{feature.label}</p>
                  <p className="text-xs text-muted-foreground">{feature.description}</p>
                </div>
              </motion.div>
            ))}
          </div>

          {/* CTA */}
          <Button
            onClick={handleActivate}
            className="w-full py-6 rounded-xl font-display font-bold text-base bg-[oklch(0.488_0.243_264.376)] hover:bg-[oklch(0.441_0.243_264.376)] text-white shadow-lg shadow-[oklch(0.488_0.243_264.376/0.3)] hover:shadow-xl hover:shadow-[oklch(0.488_0.243_264.376/0.4)] transition-all duration-300"
          >
            <span className="flex items-center gap-2">
              <Zap className="w-5 h-5" />
              Start Custom Plan — £{billingCycle === "monthly" ? "29" : "199"}/{billingCycle === "monthly" ? "mo" : "year"}
            </span>
          </Button>

          {/* Trust */}
          <p className="text-xs text-muted-foreground text-center mt-4">
            Cancel anytime · 7-day money-back guarantee · Secure payment via Stripe
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}
