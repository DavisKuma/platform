/**
 * OutreachSection — Showcases the AI email generator feature with illustration
 * Design: Warm Modernism — asymmetric layout, warm bg, email magic illustration
 */
import { Button } from "@/components/ui/button";
import { Wand2, CheckCircle2, ArrowRight } from "lucide-react";
import { motion } from "framer-motion";

const EMAIL_MAGIC_IMAGE = "https://d2xsxph8kpxj0f.cloudfront.net/310519663385886690/REzhPHfbXGKqcVNEDV6trD/email-magic-o52PTmL6wM6VPnDrLgiPC9.webp";

interface OutreachSectionProps {
  onScrollToMatches: () => void;
}

export default function OutreachSection({ onScrollToMatches }: OutreachSectionProps) {
  const benefits = [
    "Personalised subject lines that get opened",
    "Company-specific compliments and hooks",
    "Mentions your visa status as a selling point",
    "Clear call-to-action that drives replies",
  ];

  return (
    <section className="py-16 md:py-24 relative overflow-hidden">
      {/* Subtle background */}
      <div className="absolute inset-0 bg-gradient-to-b from-background via-[oklch(0.488_0.243_264.376/0.02)] to-background" />

      <div className="container relative">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-20 items-center">
          {/* Left: Illustration */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="relative order-2 lg:order-1"
          >
            <div className="relative max-w-md mx-auto lg:mx-0">
              <div className="absolute -top-6 -left-6 w-28 h-28 bg-[oklch(0.828_0.189_84.429/0.1)] rounded-full blur-2xl" />
              <div className="absolute -bottom-6 -right-6 w-36 h-36 bg-[oklch(0.488_0.243_264.376/0.08)] rounded-full blur-2xl" />
              <img
                src={EMAIL_MAGIC_IMAGE}
                alt="AI-powered email generation with magic wand and sparkles"
                className="relative rounded-3xl shadow-xl w-full"
              />
            </div>
          </motion.div>

          {/* Right: Content */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="space-y-6 order-1 lg:order-2"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[oklch(0.828_0.189_84.429/0.1)] border border-[oklch(0.828_0.189_84.429/0.2)]">
              <Wand2 className="w-4 h-4 text-[oklch(0.769_0.188_70.08)]" />
              <span className="text-sm font-medium text-[oklch(0.769_0.188_70.08)]">Outreach AI</span>
            </div>

            <h2 className="font-display font-bold text-3xl sm:text-4xl text-foreground leading-tight">
              Cold emails that feel{" "}
              <span className="text-[oklch(0.488_0.243_264.376)]">warm</span> and get{" "}
              <span className="text-[oklch(0.488_0.243_264.376)]">replies</span>
            </h2>

            <p className="text-lg text-muted-foreground leading-relaxed">
              Our AI analyses each company's profile, your CV, and the specific role to craft emails that feel genuinely personal — not template-y. Hiring managers at small companies actually read these.
            </p>

            <div className="space-y-3 py-2">
              {benefits.map((benefit, i) => (
                <motion.div
                  key={benefit}
                  initial={{ opacity: 0, x: 12 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: 0.2 + i * 0.08 }}
                  className="flex items-center gap-3"
                >
                  <CheckCircle2 className="w-5 h-5 text-[oklch(0.596_0.145_163.225)] flex-shrink-0" />
                  <span className="text-sm font-medium text-foreground">{benefit}</span>
                </motion.div>
              ))}
            </div>

            <Button
              onClick={onScrollToMatches}
              className="bg-[oklch(0.488_0.243_264.376)] hover:bg-[oklch(0.441_0.243_264.376)] text-white rounded-xl font-display font-semibold text-base px-8 py-6 shadow-lg shadow-[oklch(0.488_0.243_264.376/0.25)] hover:shadow-xl hover:shadow-[oklch(0.488_0.243_264.376/0.35)] transition-all duration-300"
            >
              Try It Now
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
