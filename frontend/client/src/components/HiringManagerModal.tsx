/**
 * HiringManagerModal — Shows locked hiring manager info behind paywall
 * Design: Warm Modernism — blurred data, indigo upgrade CTA
 */
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Lock, Zap, User, Mail, Building2, Shield } from "lucide-react";
import { motion } from "framer-motion";
import type { JobListing } from "@/lib/data";

interface HiringManagerModalProps {
  isOpen: boolean;
  onClose: () => void;
  job: JobListing | null;
  onOpenPricing: () => void;
}

export default function HiringManagerModal({ isOpen, onClose, job, onOpenPricing }: HiringManagerModalProps) {
  if (!job) return null;

  const blurredName = job.hiringManager.split("").map((c, i) => (i < 2 ? c : "•")).join("");
  const blurredEmail = job.hiringManagerEmail.split("").map((c, i) => (i < 3 ? c : "•")).join("");

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-md bg-card border-border rounded-2xl p-0">
        <div className="p-6 sm:p-8">
          <DialogHeader className="mb-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-11 h-11 rounded-xl bg-[oklch(0.488_0.243_264.376/0.1)] flex items-center justify-center">
                <User className="w-5 h-5 text-[oklch(0.488_0.243_264.376)]" />
              </div>
              <div>
                <DialogTitle className="font-display font-bold text-lg text-foreground">
                  Hiring Manager
                </DialogTitle>
                <p className="text-sm text-muted-foreground">{job.company}</p>
              </div>
            </div>
          </DialogHeader>

          {/* Company info (visible) */}
          <div className="space-y-3 mb-6">
            <div className="flex items-center gap-3 p-3 rounded-xl bg-muted/50">
              <Building2 className="w-4 h-4 text-muted-foreground" />
              <div>
                <p className="text-xs text-muted-foreground">Company</p>
                <p className="text-sm font-semibold text-foreground">{job.company}</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 rounded-xl bg-muted/50">
              <Shield className="w-4 h-4 text-[oklch(0.596_0.145_163.225)]" />
              <div>
                <p className="text-xs text-muted-foreground">Sponsor Status</p>
                <p className="text-sm font-semibold text-[oklch(0.596_0.145_163.225)]">Verified UK Visa Sponsor</p>
              </div>
            </div>
          </div>

          {/* Locked info */}
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="relative rounded-2xl border border-border overflow-hidden mb-6"
          >
            {/* Blur overlay */}
            <div className="absolute inset-0 bg-card/60 backdrop-blur-sm z-10 flex flex-col items-center justify-center">
              <div className="w-12 h-12 rounded-xl bg-[oklch(0.488_0.243_264.376/0.1)] flex items-center justify-center mb-3">
                <Lock className="w-6 h-6 text-[oklch(0.488_0.243_264.376)]" />
              </div>
              <p className="font-display font-semibold text-sm text-foreground mb-1">Upgrade to unlock</p>
              <p className="text-xs text-muted-foreground">Get verified contact details</p>
            </div>

            {/* Blurred content */}
            <div className="p-5 space-y-4 select-none">
              <div className="flex items-center gap-3">
                <User className="w-4 h-4 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">Full Name</p>
                  <p className="text-sm font-semibold text-foreground blur-[4px]">{blurredName}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Mail className="w-4 h-4 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">Verified Email</p>
                  <p className="text-sm font-semibold text-foreground blur-[4px]">{blurredEmail}</p>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Upgrade CTA */}
          <Button
            onClick={() => {
              onClose();
              onOpenPricing();
            }}
            className="w-full py-6 rounded-xl font-display font-bold text-base bg-[oklch(0.488_0.243_264.376)] hover:bg-[oklch(0.441_0.243_264.376)] text-white shadow-lg shadow-[oklch(0.488_0.243_264.376/0.3)] hover:shadow-xl hover:shadow-[oklch(0.488_0.243_264.376/0.4)] transition-all duration-300"
          >
            <Zap className="w-5 h-5 mr-2" />
            Unlock with Custom Plan — £29/mo
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
