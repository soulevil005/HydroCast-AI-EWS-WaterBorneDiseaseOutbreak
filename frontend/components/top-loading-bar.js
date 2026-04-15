"use client";

import { motion, AnimatePresence } from "framer-motion";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

export default function TopLoadingBar() {
  const pathname = usePathname();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    const timer = setTimeout(() => setLoading(false), 450);
    return () => clearTimeout(timer);
  }, [pathname]);

  return (
    <AnimatePresence>
      {loading && (
        <motion.div
          initial={{ scaleX: 0, opacity: 0.65 }}
          animate={{ scaleX: 1, opacity: 1 }}
          exit={{ opacity: 0, scaleX: 1 }}
          transition={{ duration: 0.45, ease: "easeInOut" }}
          className="fixed left-0 top-0 z-[100] h-1 w-full origin-left bg-gradient-to-r from-cyan-400 via-blue-500 to-violet-500 shadow-[0_0_18px_rgba(70,162,255,0.45)]"
        />
      )}
    </AnimatePresence>
  );
}
