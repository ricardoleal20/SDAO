# SDAO — Visual Summary

Interactive web companion to the paper
**"Stochastic diffusion adaptive optimization, a novel metaheuristic approach"**
by Ricardo M. Leal Lopez, published in *Discover Analytics* (2026) 4:6.

Live site: <https://sdao.papers.ricardoleal20.dev>

## Stack

- React 19 + TypeScript
- Vite 6 (build-time Tailwind CSS v4, self-hosted fonts)
- Recharts & HTML Canvas for the live optimization visualizations

## Develop

```bash
npm install
npm run dev      # http://localhost:3000
```

## Build

```bash
npm run build    # outputs dist/
npm run preview  # serve the production build locally
```

## Deploy

Deploys automatically to GitHub Pages on every push to `main` that touches this
folder (see `.github/workflows/deploy-ui-summary.yml`). The custom domain is
`sdao.papers.ricardoleal20.dev`.
