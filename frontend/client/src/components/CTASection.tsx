/**
 * CTASection — Bottom conversion section with career success illustration
 * Design: Warm Modernism — indigo gradient bg, warm CTA, career success image
 */
import { Button } from "@/components/ui/button";
import { Zap, ArrowRight } from "lucide-react";
import { motion } from "framer-motion";

const CAREER_IMAGE = "https://d2xsxph8kpxj0f.cloudfront.net/310519663385886690/REzhPHfbXGKqcVNEDV6trD/career-success-5vTYencRhM3PqPVYEKDaDY.webp";

interface CTASectionProps {
  onOpenPricing: () => void;
  onScrollToTop: () => void;
}

export default function CTASection({ onOpenPricing, onScrollToTop }: CTASectionProps) {
  return (
    <section className="py-16 md:py-24">
      <div className="container">
        <div className="relative rounded-3xl overflow-hidden bg-[oklch(0.488_0.243_264.376)] p-8 sm:p-12 lg:p-16">
          {/* Background decoration */}
          <div className="absolute inset-0 opacity-10">
            <div className="absolute top-0 right-0 w-96 h-96 bg-white rounded-full blur-3xl -translate-y-1/2 translate-x-1/3" />
            <div className="absolute bottom-0 left-0 w-64 h-64 bg-[oklch(0.696_0.17_162.48)] rounded-full blur-3xl translate-y-1/3 -translate-x-1/4" />
          </div>

          <div className="relative grid lg:grid-cols-[1fr_auto] gap-10 items-center">
            <div className="space-y-6">
              <motion.h2
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5 }}
                className="font-display font-bold text-3xl sm:text-4xl text-white leading-tight"
              >
                Your dream UK job is closer than you think
              </motion.h2>

              <motion.p
                initial={{ opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.1 }}
                className="text-lg text-white/80 max-w-xl leading-relaxed"
              >
                Join 2,400+ international students who've already found visa-sponsored roles at small UK companies. Upload your CV and start sending personalised outreach today.
              </motion.p>

              <motion.div
                initial={{ opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="flex flex-col sm:flex-row gap-3"
              >
                <Button
                  onClick={onScrollToTop}
                  className="bg-white text-[oklch(0.488_0.243_264.376)] hover:bg-white/90 rounded-xl font-display font-bold text-base px-8 py-6 shadow-lg transition-all duration-300"
                >
                  Get Started Free
                  <ArrowRight className="w-5 h-5 ml-2" />
                </Button>
                <Button
                  onClick={onOpenPricing}
                  variant="outline"
                  className="border-white/30 bg-transparent text-white hover:bg-white/10 rounded-xl font-display font-semibold text-base px-8 py-6 transition-all duration-300"
                >
                  <Zap className="w-5 h-5 mr-2" />
                  View Custom Plan
                </Button>
              </motion.div>
            </div>

            {/* Career success image */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="hidden lg:block"
            >
              <img
                src={CAREER_IMAGE}
                alt="Student celebrating job offer with visa approval"
                className="w-56 h-auto rounded-2xl shadow-2xl shadow-black/20"
              />
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
}
