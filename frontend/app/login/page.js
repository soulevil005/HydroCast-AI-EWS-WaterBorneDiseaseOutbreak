"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useMemo, useState } from "react";
import { useAuth } from "../../context/AuthContext";

const tabs = ["Login", "Create Account"];

export default function LoginPage() {
  const { login, signup } = useAuth();
  const [activeTab, setActiveTab] = useState("Login");
  const [remember, setRemember] = useState(true);
  const [loginForm, setLoginForm] = useState({ email: "demo@hydrocast.ai", password: "demo123" });
  const [signupForm, setSignupForm] = useState({ name: "", email: "", password: "" });
  const [submitting, setSubmitting] = useState(false);

  const title = useMemo(
    () =>
      activeTab === "Login"
        ? "Access the HydroCast command center"
        : "Create a controlled HydroCast access account",
    [activeTab],
  );

  async function handleLogin(event) {
    event.preventDefault();
    setSubmitting(true);
    await login({ ...loginForm, remember });
    setSubmitting(false);
  }

  async function handleSignup(event) {
    event.preventDefault();
    setSubmitting(true);
    const result = await signup(signupForm);
    setSubmitting(false);
    if (result.ok) {
      setActiveTab("Login");
      setLoginForm({ email: signupForm.email, password: signupForm.password });
      setSignupForm({ name: "", email: "", password: "" });
    }
  }

  return (
    <main className="min-h-screen bg-command px-4 py-6 text-white lg:px-6">
      <div className="mx-auto grid min-h-[calc(100vh-3rem)] max-w-[1600px] overflow-hidden rounded-[2rem] border border-white/10 glass-panel lg:grid-cols-[1.08fr_0.92fr]">
        <section className="relative flex flex-col justify-between overflow-hidden border-b border-white/10 p-8 lg:border-b-0 lg:border-r lg:p-12">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(70,162,255,0.22),transparent_24%),radial-gradient(circle_at_80%_0%,rgba(171,140,255,0.18),transparent_28%),radial-gradient(circle_at_80%_80%,rgba(46,211,154,0.14),transparent_22%)]" />
          <div className="relative">
            <div className="text-[11px] font-extrabold uppercase tracking-[0.28em] text-slate-400">AI-Powered Waterborne Disease Intelligence</div>
            <div className="orbitron mt-6 text-5xl font-bold leading-none md:text-7xl">HydroCast</div>
            <div className="mt-5 max-w-xl text-2xl font-bold text-slate-100">Premium outbreak intelligence for faster public-health action.</div>
            <div className="mt-6 max-w-2xl text-base leading-8 text-slate-300">
              HydroCast fuses epidemiological forecasts, district risk mapping, SHAP explainability, and response readiness into one command-center
              workflow for Maharashtra.
            </div>
          </div>

          <div className="relative mt-8 grid gap-4 md:grid-cols-3">
            {[
              ["36", "Districts monitored"],
              ["0.891", "Model macro F1"],
              ["Live", "API-backed intelligence"],
            ].map(([value, label]) => (
              <motion.div key={label} whileHover={{ y: -6 }} className="rounded-[1.4rem] border border-white/10 bg-white/[0.04] p-5 backdrop-blur-xl">
                <div className="text-3xl font-black text-white">{value}</div>
                <div className="mt-2 text-sm text-slate-300">{label}</div>
              </motion.div>
            ))}
          </div>
        </section>

        <section className="flex items-center justify-center p-6 lg:p-10">
          <div className="w-full max-w-xl rounded-[1.8rem] border border-white/10 bg-white/[0.05] p-6 shadow-panel backdrop-blur-2xl lg:p-8">
            <div className="flex gap-2 rounded-full border border-white/10 bg-white/[0.04] p-1">
              {tabs.map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`flex-1 rounded-full px-4 py-3 text-sm font-bold transition ${
                    activeTab === tab ? "bg-gradient-to-r from-cyan-400 to-violet-500 text-slate-950" : "text-slate-300"
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            <div className="mt-6">
              <div className="text-3xl font-black text-white">{title}</div>
              <div className="mt-2 text-sm leading-7 text-slate-300">
                Demo user always works: <span className="font-bold text-cyan-300">demo@hydrocast.ai</span> /{" "}
                <span className="font-bold text-cyan-300">demo123</span>
              </div>
            </div>

            <AnimatePresence mode="wait">
              {activeTab === "Login" ? (
                <motion.form key="login" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} className="mt-8 space-y-5" onSubmit={handleLogin}>
                  <AuthInput label="Email" type="email" value={loginForm.email} onChange={(value) => setLoginForm((prev) => ({ ...prev, email: value }))} />
                  <AuthInput label="Password" type="password" value={loginForm.password} onChange={(value) => setLoginForm((prev) => ({ ...prev, password: value }))} />

                  <div className="flex items-center justify-between text-sm">
                    <label className="flex items-center gap-3 text-slate-300">
                      <input type="checkbox" checked={remember} onChange={(event) => setRemember(event.target.checked)} className="h-4 w-4 rounded border-white/20 bg-transparent accent-cyan-400" />
                      Remember me
                    </label>
                    <button type="button" className="font-semibold text-cyan-300">Forgot password?</button>
                  </div>

                  <AuthButton label={submitting ? "Signing in..." : "Login to HydroCast"} />
                  <SecondaryButton label="Continue with Google" />
                </motion.form>
              ) : (
                <motion.form key="signup" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} className="mt-8 space-y-5" onSubmit={handleSignup}>
                  <AuthInput label="Name" value={signupForm.name} onChange={(value) => setSignupForm((prev) => ({ ...prev, name: value }))} />
                  <AuthInput label="Email" type="email" value={signupForm.email} onChange={(value) => setSignupForm((prev) => ({ ...prev, email: value }))} />
                  <AuthInput label="Password" type="password" value={signupForm.password} onChange={(value) => setSignupForm((prev) => ({ ...prev, password: value }))} />
                  <div className="rounded-[1.2rem] border border-white/10 bg-white/[0.04] px-4 py-3 text-sm leading-7 text-slate-300">
                    Password must be at least 6 characters and email must be unique across HydroCast local accounts.
                  </div>
                  <AuthButton label={submitting ? "Creating account..." : "Create HydroCast account"} />
                </motion.form>
              )}
            </AnimatePresence>
          </div>
        </section>
      </div>
    </main>
  );
}

function AuthInput({ label, type = "text", value, onChange }) {
  return (
    <motion.label whileFocus={{ scale: 1.01 }} className="block">
      <div className="mb-2 text-sm font-semibold text-slate-300">{label}</div>
      <input
        type={type}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded-[1.15rem] border border-white/10 bg-slate-950/40 px-4 py-3.5 text-white outline-none transition focus:border-cyan-400/60 focus:shadow-[0_0_24px_rgba(34,211,238,0.15)]"
        required
      />
    </motion.label>
  );
}

function AuthButton({ label }) {
  return (
    <motion.button
      whileHover={{ y: -2, scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
      type="submit"
      className="w-full rounded-[1.2rem] bg-gradient-to-r from-cyan-400 via-blue-500 to-violet-500 px-4 py-3.5 text-base font-black text-slate-950 shadow-[0_16px_40px_rgba(70,162,255,0.25)]"
    >
      {label}
    </motion.button>
  );
}

function SecondaryButton({ label }) {
  return (
    <motion.button
      whileHover={{ y: -2 }}
      type="button"
      className="w-full rounded-[1.2rem] border border-white/10 bg-white/[0.04] px-4 py-3.5 text-base font-bold text-white"
    >
      {label}
    </motion.button>
  );
}
