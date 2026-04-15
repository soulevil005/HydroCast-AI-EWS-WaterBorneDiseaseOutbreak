/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,jsx}",
    "./components/**/*.{js,jsx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#040913",
        panel: "#0b1627",
        line: "rgba(146, 165, 199, 0.16)",
        critical: "#ff536f",
        high: "#ffb24c",
        info: "#46a2ff",
        safe: "#2ed39a",
        ai: "#ab8cff"
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(70,162,255,0.2), 0 18px 45px rgba(3,8,16,0.38)",
        panel: "0 24px 60px rgba(0,0,0,0.34)"
      },
      borderRadius: {
        xl2: "1.1rem"
      },
      backgroundImage: {
        command: "radial-gradient(circle at top left, rgba(70,162,255,0.14), transparent 26%), radial-gradient(circle at top right, rgba(171,140,255,0.16), transparent 24%), radial-gradient(circle at bottom left, rgba(46,211,154,0.1), transparent 22%), linear-gradient(180deg, #040913 0%, #07101d 48%, #03070e 100%)"
      }
    },
  },
  plugins: [],
};
