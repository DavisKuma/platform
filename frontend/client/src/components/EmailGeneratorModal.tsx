/**
 * EmailGeneratorModal — AI-powered cold email generator with typewriter effect
 * Generates a template email based on job details (no backend required).
 */
import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Copy, RefreshCw, Mail, Check, Sparkles, Wand2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { toast } from "sonner";
import type { JobListing } from "@/lib/data";

interface EmailGeneratorModalProps {
  isOpen: boolean;
  onClose: () => void;
  job: JobListing | null;
  onOpenPricing?: () => void;
}

function generateTemplate(job: JobListing) {
  const subject = `Application: ${job.jobTitle} at ${job.company}`;
  const body = `Dear Hiring Manager,

I am writing to express my strong interest in the ${job.jobTitle} position at ${job.company}${job.location ? ` in ${job.location}` : ""}.

Having reviewed the role requirements, I believe my skills and experience make me an excellent candidate for this opportunity.${job.fitReason ? `\n\n${job.fitReason}` : ""}

I would welcome the opportunity to discuss how my background aligns with ${job.company}'s needs and how I can contribute to the team's success.

Thank you for your time and consideration. I look forward to hearing from you.

Best regards`;
  return { subject, body };
}

export default function EmailGeneratorModal({ isOpen, onClose, job }: EmailGeneratorModalProps) {
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [displayedBody, setDisplayedBody] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [copied, setCopied] = useState(false);
  const [hasGenerated, setHasGenerated] = useState(false);

  const generate = useCallback(async () => {
    if (!job) return;

    setIsGenerating(true);
    setDisplayedBody("");
    setIsTyping(false);
    setHasGenerated(false);

    // Small delay for UX
    await new Promise((r) => setTimeout(r, 800));

    const result = generateTemplate(job);
    setSubject(result.subject);
    setBody(result.body);
    setIsGenerating(false);
    setHasGenerated(true);
    setIsTyping(true);
  }, [job]);

  // Typewriter effect
  useEffect(() => {
    if (!isTyping || !body) return;

    const fullBody = body;
    let currentIndex = 0;

    const interval = setInterval(() => {
      if (currentIndex < fullBody.length) {
        const charsToAdd = Math.min(3, fullBody.length - currentIndex);
        setDisplayedBody(fullBody.slice(0, currentIndex + charsToAdd));
        currentIndex += charsToAdd;
      } else {
        clearInterval(interval);
        setIsTyping(false);
      }
    }, 12);

    return () => clearInterval(interval);
  }, [isTyping, body]);

  // Auto-generate on open
  useEffect(() => {
    if (isOpen && job) {
      generate();
    }
    if (!isOpen) {
      setSubject("");
      setBody("");
      setDisplayedBody("");
      setIsTyping(false);
      setCopied(false);
      setHasGenerated(false);
    }
  }, [isOpen, job]);

  const handleCopy = () => {
    if (!body) return;
    navigator.clipboard.writeText(`Subject: ${subject}\n\n${body}`);
    setCopied(true);
    toast.success("Email copied to clipboard!");
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSendViaGmail = () => {
    if (!body) return;
    const gmailUrl = `https://mail.google.com/mail/?view=cm&fs=1&su=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
    window.open(gmailUrl, "_blank");
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto bg-card border-border rounded-2xl p-0">
        {/* Header */}
        <div className="sticky top-0 z-10 bg-card border-b border-border rounded-t-2xl">
          <DialogHeader className="p-6 pb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-[oklch(0.488_0.243_264.376/0.1)] flex items-center justify-center">
                <Wand2 className="w-5 h-5 text-[oklch(0.488_0.243_264.376)]" />
              </div>
              <div className="flex-1">
                <DialogTitle className="font-display font-bold text-lg text-foreground">
                  AI Email Generator
                </DialogTitle>
                {job && (
                  <p className="text-sm text-muted-foreground mt-0.5">
                    Outreach to {job.company}
                  </p>
                )}
              </div>
            </div>
          </DialogHeader>
        </div>

        <div className="p-6 pt-4">
          <AnimatePresence mode="wait">
            {isGenerating ? (
              <motion.div
                key="generating"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center justify-center py-16 space-y-6"
              >
                <div className="relative">
                  <div className="w-20 h-20 rounded-2xl bg-[oklch(0.488_0.243_264.376/0.1)] flex items-center justify-center">
                    <Sparkles className="w-10 h-10 text-[oklch(0.488_0.243_264.376)] animate-pulse" />
                  </div>
                </div>
                <div className="text-center">
                  <p className="font-display font-semibold text-foreground mb-1">Crafting your personalised email...</p>
                  <p className="text-sm text-muted-foreground">
                    Generating outreach for {job?.company}
                  </p>
                </div>
                <div className="w-48 h-1.5 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-[oklch(0.488_0.243_264.376)] rounded-full animate-shimmer" style={{ width: "60%" }} />
                </div>
              </motion.div>
            ) : hasGenerated ? (
              <motion.div
                key="email"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
                className="space-y-5"
              >
                {/* Subject line */}
                <div className="bg-muted/50 rounded-xl p-4 border border-border">
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5">Subject</p>
                  <p className="font-display font-semibold text-foreground">{subject}</p>
                </div>

                {/* Email body */}
                <div className="bg-card rounded-xl p-5 border border-border min-h-[300px]">
                  <pre className="whitespace-pre-wrap font-sans text-sm text-foreground leading-relaxed">
                    {displayedBody}
                    {isTyping && (
                      <span className="inline-block w-0.5 h-4 bg-[oklch(0.488_0.243_264.376)] ml-0.5 animate-pulse" />
                    )}
                  </pre>
                </div>

                {/* Action buttons */}
                <div className="flex flex-col sm:flex-row gap-3">
                  <Button
                    onClick={handleCopy}
                    variant="outline"
                    className="flex-1 rounded-xl font-medium border-border hover:border-[oklch(0.488_0.243_264.376/0.3)] transition-all duration-200"
                    disabled={isTyping}
                  >
                    {copied ? (
                      <>
                        <Check className="w-4 h-4 mr-2 text-[oklch(0.596_0.145_163.225)]" />
                        Copied!
                      </>
                    ) : (
                      <>
                        <Copy className="w-4 h-4 mr-2" />
                        Copy to Clipboard
                      </>
                    )}
                  </Button>

                  <Button
                    onClick={() => generate()}
                    variant="outline"
                    className="flex-1 rounded-xl font-medium border-border hover:border-[oklch(0.828_0.189_84.429/0.3)] transition-all duration-200"
                    disabled={isTyping || isGenerating}
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Regenerate
                  </Button>

                  <Button
                    onClick={handleSendViaGmail}
                    className="flex-1 rounded-xl font-semibold bg-[oklch(0.488_0.243_264.376)] hover:bg-[oklch(0.441_0.243_264.376)] text-white shadow-md shadow-[oklch(0.488_0.243_264.376/0.2)] transition-all duration-300"
                    disabled={isTyping}
                  >
                    <Mail className="w-4 h-4 mr-2" />
                    Send via Gmail
                  </Button>
                </div>

                {/* Disclaimer */}
                <p className="text-xs text-muted-foreground pt-2">
                  Template email — review and personalise before sending.
                </p>
              </motion.div>
            ) : null}
          </AnimatePresence>
        </div>
      </DialogContent>
    </Dialog>
  );
}
