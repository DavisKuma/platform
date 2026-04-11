/**
 * StatsSection — Key platform metrics in a warm, trust-building layout
 * Design: Warm Modernism — indigo bg block, warm stat cards
 */
import { motion } from "framer-motion";
import { Building2, Users, Mail, TrendingUp } from "lucide-react";

export default function StatsSection() {
  const stats = [
    {
      icon: Building2,
      value: "121,000+",
      label: "Verified Sponsors",
      description: "Companies with active UK visa licences",
      iconBg: "bg-[oklch(0.488_0.243_264.376/0.1)]",
      iconColor: "text-[oklch(0.488_0.243_264.376)]",
      cornerBg: "bg-[oklch(0.488_0.243_264.376)]",
    },
    {
      icon: Users,
      value: "≤100",
      label: "Max Employees",
      description: "Small companies that actually respond",
      iconBg: "bg-[oklch(0.696_0.17_162.48/0.1)]",
      iconColor: "text-[oklch(0.696_0.17_162.48)]",
      cornerBg: "bg-[oklch(0.696_0.17_162.48)]",
    },
    {
      icon: Mail,
      value: "34%",
      label: "Reply Rate",
      description: "Average for AI-generated outreach",
      iconBg: "bg-[oklch(0.828_0.189_84.429/0.1)]",
      iconColor: "text-[oklch(0.828_0.189_84.429)]",
      cornerBg: "bg-[oklch(0.828_0.189_84.429)]",
    },
    {
      icon: TrendingUp,
      value: "12 days",
      label: "Avg. Time to Interview",
      description: "From first email to interview invite",
      iconBg: "bg-[oklch(0.541_0.281_275.847/0.1)]",
      iconColor: "text-[oklch(0.541_0.281_275.847)]",
      cornerBg: "bg-[oklch(0.541_0.281_275.847)]",
    },
  ];

  return (
    <section className="py-16 md:py-20">
      <div className="container">
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {stats.map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.08 }}
              className="relative bg-card rounded-2xl border border-border p-6 hover:shadow-lg hover:shadow-[oklch(0.488_0.243_264.376/0.05)] transition-all duration-300 group overflow-hidden"
            >
              {/* Decorative corner */}
              <div
                className={`absolute top-0 right-0 w-24 h-24 rounded-bl-[3rem] opacity-[0.04] group-hover:opacity-[0.08] transition-opacity duration-300 ${stat.cornerBg}`}
              />

              <div className={`w-11 h-11 rounded-xl flex items-center justify-center mb-4 ${stat.iconBg}`}>
                <stat.icon className={`w-5 h-5 ${stat.iconColor}`} />
              </div>

              <p className="font-display font-extrabold text-3xl text-foreground mb-1">{stat.value}</p>
              <p className="font-display font-semibold text-sm text-foreground mb-1">{stat.label}</p>
              <p className="text-xs text-muted-foreground">{stat.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
