/**
 * Dashboard — User dashboard with overview stats
 * Note: Saved matches and outreach features require backend endpoints (coming soon).
 */
import { useAuth } from "@/_core/hooks/useAuth";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Bookmark, Mail, Zap, FileText, ArrowRight, Crown } from "lucide-react";
import { motion } from "framer-motion";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export default function Dashboard() {
  const { user } = useAuth();
  const [, navigate] = useLocation();

  const savedCount = 0;
  const outreachCount = 0;
  const isPaid = user?.subscriptionTier !== "free";

  const stats = [
    { label: "Saved Matches", value: savedCount, icon: Bookmark, color: "oklch(0.488_0.243_264.376)", href: "/matches" },
    { label: "Emails Generated", value: outreachCount, icon: Mail, color: "oklch(0.596_0.145_163.225)", href: "/outreach" },
    { label: "Emails Today", value: "0/3", icon: Zap, color: "oklch(0.828_0.189_84.429)", href: "#" },
    { label: "Plan", value: isPaid ? "Custom" : "Free", icon: isPaid ? Crown : FileText, color: isPaid ? "oklch(0.696_0.17_162.48)" : "oklch(0.6_0.1_250)", href: "#" },
  ];

  return (
    <div className="min-h-screen bg-background">
      <Navbar activeSection="dashboard" onNavigate={() => {}} onOpenPricing={() => {}} />

      <div className="container max-w-6xl mx-auto px-4 pt-28 pb-16">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="mb-10">
          <h1 className="font-display font-bold text-3xl text-foreground mb-2">
            Welcome back{user?.name ? `, ${user.name.split(" ")[0]}` : ""} 👋
          </h1>
          <p className="text-muted-foreground">
            Here's your job search overview. Upload your CV on the home page to get started.
          </p>
        </motion.div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
          {stats.map((stat, i) => (
            <motion.div key={stat.label} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.08 }}>
              <Card
                className="cursor-pointer hover:shadow-md transition-all duration-200 border-border bg-card"
                onClick={() => stat.href !== "#" && navigate(stat.href)}
              >
                <CardContent className="p-5">
                  <div className="flex items-center justify-between mb-3">
                    <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ backgroundColor: `${stat.color}15` }}>
                      <stat.icon className="w-5 h-5" style={{ color: stat.color }} />
                    </div>
                  </div>
                  <p className="font-display font-bold text-2xl text-foreground">{stat.value}</p>
                  <p className="text-sm text-muted-foreground mt-0.5">{stat.label}</p>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Quick Actions */}
        <div className="grid md:grid-cols-2 gap-6 mb-10">
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
            <Card className="border-border bg-card h-full">
              <CardHeader>
                <CardTitle className="font-display text-lg flex items-center gap-2">
                  <Bookmark className="w-5 h-5 text-[oklch(0.488_0.243_264.376)]" />
                  Recent Saved Matches
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8">
                  <p className="text-muted-foreground text-sm mb-3">No saved matches yet</p>
                  <Button variant="outline" size="sm" className="rounded-xl" onClick={() => navigate("/")}>
                    Browse Jobs <ArrowRight className="w-4 h-4 ml-1" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
            <Card className="border-border bg-card h-full">
              <CardHeader>
                <CardTitle className="font-display text-lg flex items-center gap-2">
                  <Mail className="w-5 h-5 text-[oklch(0.596_0.145_163.225)]" />
                  Recent Outreach
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8">
                  <p className="text-muted-foreground text-sm mb-3">No emails generated yet</p>
                  <Button variant="outline" size="sm" className="rounded-xl" onClick={() => navigate("/")}>
                    Upload CV to Get Started <ArrowRight className="w-4 h-4 ml-1" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Upgrade CTA for free users */}
        {!isPaid && (
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
            <Card className="border-[oklch(0.488_0.243_264.376/0.2)] bg-[oklch(0.488_0.243_264.376/0.03)]">
              <CardContent className="p-6 flex flex-col sm:flex-row items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-[oklch(0.488_0.243_264.376/0.1)] flex items-center justify-center">
                    <Crown className="w-6 h-6 text-[oklch(0.488_0.243_264.376)]" />
                  </div>
                  <div>
                    <p className="font-display font-bold text-foreground">Upgrade to Custom Plan</p>
                    <p className="text-sm text-muted-foreground">Unlimited emails, hiring manager contacts, and priority matching</p>
                  </div>
                </div>
                <Button
                  className="bg-[oklch(0.488_0.243_264.376)] hover:bg-[oklch(0.441_0.243_264.376)] text-white rounded-xl font-display font-semibold px-6 whitespace-nowrap"
                  onClick={() => navigate("/?pricing=open")}
                >
                  <Zap className="w-4 h-4 mr-2" /> From £29/mo
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </div>

      <Footer />
    </div>
  );
}
