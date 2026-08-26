/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { HeroSimulation } from './components/HeroSimulation';
import { TheoreticalFoundations } from './components/TheoreticalFoundations';
import { AdaptiveMechanics } from './components/AdaptiveMechanics';
import { BenchmarkResults } from './components/BenchmarkResults';
import { AblationStudy } from './components/AblationStudy';
import { SDAOSimulator } from './components/SDAOSimulator';
import { Reveal } from './components/Reveal';
import {
  Menu, X, Github, ExternalLink, ArrowRight, ArrowDown,
} from 'lucide-react';

const NAV_ITEMS = [
  { id: 'simulator', label: 'Simulator' },
  { id: 'theory', label: 'Theory' },
  { id: 'mechanics', label: 'Engine' },
  { id: 'results', label: 'Results' },
  { id: 'ablation', label: 'Ablation' },
];

const HeroStats = [
  { value: 'd = 500', label: 'SOCO11 large-scale scalability', sub: 'mean-square stable to 500 dimensions' },
  { value: '38 / 38', label: 'CEC 2017 Wilcoxon wins', sub: 'statistically significant dominance' },
  { value: 'p < 10⁻⁴⁰', label: 'One-way ANOVA power', sub: 'across 30 independent runs' },
  { value: '10.7×', label: 'Lower error vs. SFS at d=100', sub: 'on stochastic benchmarks' },
];

const Bento = [
  {
    span: 'lg:col-span-3 lg:row-span-2',
    kicker: 'Density-driven diffusion',
    title: 'Escape every local basin',
    body: 'A Fickian repulsion field (D_FL) pushes candidate solutions away from overcrowded optima, while Brownian noise keeps the swarm probing uncharted territory — the precise behaviour standard gradient descent cannot reproduce.',
    accent: true,
  },
  {
    span: 'lg:col-span-2',
    kicker: 'Adaptive coefficients',
    title: 'α(k), γ(k), D(k) self-tune',
    body: 'Exponential decay of the learning rate, memory-driven intensification, and a dimension-aware diffusion cap regulate exploration-to-exploitation across the entire optimization horizon.',
  },
  {
    span: 'lg:col-span-2',
    kicker: 'Opposition-based learning',
    title: 'OBL breaks stagnation',
    body: 'Probabilistic reflection of the worst solutions under sustained stagnation — the ablation shows removing it inflates error by an order of magnitude.',
  },
  {
    span: 'lg:col-span-2',
    kicker: 'Convergence guarantee',
    title: 'Mean-square stable',
    body: 'One-sided Lipschitz dissipativity and a 1/(2n) variance ceiling give classical strong convergence of order 1/2 — proven, not empirical.',
  },
  {
    span: 'lg:col-span-3',
    kicker: 'Real-world & stochastic',
    title: 'Noise is not an obstacle',
    body: 'Benchmarked on six industrial decision problems and a stochastic suite with additive Gaussian noise, SDAO consistently outperforms eleven state-of-the-art metaheuristics where gradient methods diverge.',
  },
];

const AuthorCard: React.FC = () => {
  const [copied, setCopied] = useState(false);

  const bibtex = `@article{leal2026sdao,
  title={Stochastic diffusion adaptive optimization, a novel metaheuristic approach},
  author={Leal Lopez, Ricardo M.},
  journal={Discover Analytics},
  volume={4},
  number={6},
  year={2026},
  doi={10.1007/s44257-025-00054-1}
}`;

  const handleCopy = () => {
    navigator.clipboard.writeText(bibtex);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Reveal className="mx-auto max-w-5xl">
      <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-8 md:p-12 backdrop-blur-sm">
        <div className="grid grid-cols-1 gap-10 md:grid-cols-12">
          <div className="md:col-span-5">
            <span className="eyebrow">Corresponding author</span>
            <h3 className="display mt-3 text-3xl md:text-4xl text-bone">Ricardo M. Leal Lopez</h3>
            <p className="mt-1 text-sm font-semibold uppercase tracking-[0.18em] text-gold">Independent Researcher</p>
            <p className="mt-5 text-sm leading-relaxed text-white/60">
              Published in <strong className="text-white/80">Discover Analytics (2026) 4:6</strong> under a
              Creative Commons Attribution 4.0 International License.
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <a
                href="https://github.com/ricardoleal20/SDAO"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 rounded-full bg-bone px-5 py-2.5 text-xs font-bold uppercase tracking-wider text-ink transition-transform hover:scale-[1.03]"
              >
                <Github size={15} /> Repository
              </a>
              <a
                href="https://doi.org/10.1007/s44257-025-00054-1"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 rounded-full border border-white/20 px-5 py-2.5 text-xs font-bold uppercase tracking-wider text-bone transition-colors hover:border-gold hover:text-gold"
              >
                <ExternalLink size={15} /> Read paper
              </a>
            </div>
          </div>

          <div className="md:col-span-7">
            <div className="rounded-xl border border-white/10 bg-ink/60 p-6">
              <div className="flex items-center justify-between border-b border-white/10 pb-3">
                <span className="font-mono text-xs font-bold text-gold">BibTeX citation</span>
                <button
                  onClick={handleCopy}
                  className="rounded-md border border-white/15 px-3 py-1 font-mono text-xs text-white/80 transition-colors hover:border-gold hover:text-gold"
                >
                  {copied ? 'Copied' : 'Copy'}
                </button>
              </div>
              <pre className="mt-4 overflow-x-auto whitespace-pre font-mono text-xs leading-relaxed text-white/70">{bibtex}</pre>
            </div>
          </div>
        </div>
      </div>
    </Reveal>
  );
};

const App: React.FC = () => {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (id: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    setMenuOpen(false);
    const el = document.getElementById(id);
    if (el) {
      window.scrollTo({ top: el.getBoundingClientRect().top + window.pageYOffset - 84, behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen overflow-x-hidden">
      {/* Navigation */}
      <nav
        className={`fixed inset-x-0 top-0 z-50 transition-all duration-500 ${
          scrolled ? 'bg-ink/85 py-3 shadow-[0_1px_0_0_rgba(255,255,255,0.08)] backdrop-blur-xl' : 'bg-transparent py-6'
        }`}
      >
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6">
          <button
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="flex items-baseline gap-2"
          >
            <span className="display text-xl text-bone">SDAO</span>
            <span className="font-mono text-[10px] uppercase tracking-[0.3em] text-gold">2026</span>
          </button>

          <div className="hidden items-center gap-8 lg:flex">
            {NAV_ITEMS.map((item) => (
              <a
                key={item.id}
                href={`#${item.id}`}
                onClick={scrollToSection(item.id)}
                className="text-xs font-semibold uppercase tracking-[0.15em] text-white/55 transition-colors hover:text-bone"
              >
                {item.label}
              </a>
            ))}
            <a
              href="https://github.com/ricardoleal20/SDAO"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-full border border-white/20 px-4 py-2 text-xs font-bold uppercase tracking-wider text-bone transition-colors hover:border-gold hover:text-gold"
            >
              <Github size={14} /> GitHub
            </a>
          </div>

          <button
            className="text-bone lg:hidden"
            onClick={() => setMenuOpen(!menuOpen)}
            aria-label="Toggle menu"
          >
            {menuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </nav>

      {/* Mobile menu */}
      {menuOpen && (
        <div className="fixed inset-0 z-40 flex flex-col items-center justify-center gap-6 bg-ink p-6">
          {NAV_ITEMS.map((item) => (
            <a
              key={item.id}
              href={`#${item.id}`}
              onClick={scrollToSection(item.id)}
              className="display text-2xl text-bone"
            >
              {item.label}
            </a>
          ))}
          <a
            href="https://github.com/ricardoleal20/SDAO"
            target="_blank"
            rel="noopener noreferrer"
            onClick={() => setMenuOpen(false)}
            className="mt-4 inline-flex items-center gap-2 rounded-full bg-bone px-6 py-3 text-sm font-bold uppercase tracking-wider text-ink"
          >
            <Github size={16} /> View repository
          </a>
        </div>
      )}

      {/* Hero */}
      <header className="grain relative flex min-h-screen items-center overflow-hidden px-6 pt-32 pb-20">
        <div
          className="pointer-events-none absolute left-1/2 top-1/3 h-[700px] w-[700px] -translate-x-1/2 -translate-y-1/2 rounded-full opacity-25 blur-[140px]"
          style={{ background: 'radial-gradient(circle, var(--color-gold) 0%, transparent 70%)' }}
        />
        <div className="relative z-10 mx-auto max-w-6xl">
          <Reveal className="mb-7">
            <span className="inline-flex items-center gap-2.5 rounded-full border border-gold/40 bg-gold/5 px-4 py-1.5 text-[11px] font-semibold uppercase tracking-[0.2em] text-gold-soft">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-gold" />
              Discover Analytics (2026) 4:6 — Open Access
            </span>
          </Reveal>

          <Reveal delay={120}>
            <h1 className="display max-w-6xl text-[clamp(2.75rem,7vw,6rem)] text-bone">
              Stochastic Diffusion
              <br />
              Adaptive Optimization
            </h1>
          </Reveal>

          <Reveal delay={240} className="mt-7 max-w-2xl">
            <p className="font-serif text-xl italic leading-snug text-white/70 md:text-2xl">
              A novel metaheuristic grounded in diffusion dynamics, Fick&rsquo;s second law, and stochastic modeling.
            </p>
          </Reveal>

          <Reveal delay={360} className="mt-6 max-w-2xl">
            <p className="text-base leading-relaxed text-white/55">
              By replacing traditional gradient descent with a density-driven diffusion mechanism (D_FL), SDAO repels
              candidate solutions from overcrowded local optima basins and conquers noisy, deceptive, and
              high-dimensional search spaces up to d = 500.
            </p>
          </Reveal>

          <Reveal delay={480} className="mt-10 flex flex-wrap items-center gap-4">
            <a
              href="#simulator"
              onClick={scrollToSection('simulator')}
              className="inline-flex items-center gap-2 rounded-full bg-gold px-7 py-3.5 text-xs font-bold uppercase tracking-wider text-ink transition-transform hover:scale-[1.03]"
            >
              Launch live simulator <ArrowRight size={15} />
            </a>
            <a
              href="https://doi.org/10.1007/s44257-025-00054-1"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-full border border-white/20 px-7 py-3.5 text-xs font-bold uppercase tracking-wider text-bone transition-colors hover:border-gold hover:text-gold"
            >
              Read the paper <ExternalLink size={15} />
            </a>
          </Reveal>

          <a
            href="#simulator"
            onClick={scrollToSection('simulator')}
            className="mt-20 inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-white/40 transition-colors hover:text-gold"
          >
            <ArrowDown size={14} className="scroll-cue" /> Explore
          </a>
        </div>
      </header>

      {/* Interest — hero metrics marquee */}
      <section className="border-y border-white/10 bg-white/[0.02] py-5">
        <div className="flex overflow-hidden">
          <div className="marquee-track flex shrink-0 items-center gap-12 pr-12">
            {[...HeroStats, ...HeroStats, ...HeroStats].map((s, i) => (
              <span key={i} className="flex items-center gap-3 whitespace-nowrap">
                <span className="display text-2xl text-gold">{s.value}</span>
                <span className="text-xs font-semibold uppercase tracking-wider text-white/45">{s.label}</span>
                <span className="text-white/15">/</span>
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* Interest — gapless bento */}
      <section className="mx-auto max-w-7xl px-6 py-28 md:py-40">
        <Reveal className="mb-16 max-w-3xl">
          <span className="eyebrow">The mechanism</span>
          <h2 className="display mt-4 text-4xl text-bone md:text-5xl">
            Physics, not heuristics, drives the search.
          </h2>
        </Reveal>

        <div className="grid grid-flow-dense grid-cols-1 gap-4 lg:grid-cols-6 lg:grid-rows-2">
          {Bento.map((card, i) => (
            <Reveal key={i} delay={i * 80} className={card.span}>
              <article
                className={`h-full rounded-2xl border p-7 transition-colors duration-500 hover:border-gold/40 ${
                  card.accent
                    ? 'border-gold/30 bg-gradient-to-br from-gold/10 to-transparent'
                    : 'border-white/10 bg-white/[0.02]'
                }`}
              >
                <span className="eyebrow">{card.kicker}</span>
                <h3 className="display mt-3 text-2xl text-bone">{card.title}</h3>
                <p className="mt-3 text-sm leading-relaxed text-white/55">{card.body}</p>
              </article>
            </Reveal>
          ))}
        </div>
      </section>

      {/* Desire — interactive content assembly */}
      <main className="mx-auto max-w-7xl px-6">
        <section id="hero-sim" className="py-16 md:py-24">
          <HeroSimulation />
        </section>
        <section id="simulator" className="py-16 md:py-24">
          <SDAOSimulator />
        </section>
        <section id="theory" className="py-16 md:py-24">
          <TheoreticalFoundations />
        </section>
        <section id="mechanics" className="py-16 md:py-24">
          <AdaptiveMechanics />
        </section>
        <section id="results" className="py-16 md:py-24">
          <BenchmarkResults />
        </section>
        <section id="ablation" className="py-16 md:py-24">
          <AblationStudy />
        </section>
        <section id="author" className="py-16 md:py-28">
          <AuthorCard />
        </section>
      </main>

      {/* Action — footer */}
      <footer className="border-t border-white/10 bg-ink px-6 py-24">
        <div className="mx-auto max-w-7xl">
          <div className="flex flex-col gap-12 md:flex-row md:items-end md:justify-between">
            <div className="max-w-xl">
              <span className="eyebrow">Cite & build on it</span>
              <h2 className="display mt-4 text-4xl text-bone md:text-5xl">
                The code is open.
                <br />
                The proof is reproducible.
              </h2>
              <div className="mt-7 flex flex-wrap gap-4">
                <a
                  href="https://github.com/ricardoleal20/SDAO"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 rounded-full bg-bone px-7 py-3.5 text-xs font-bold uppercase tracking-wider text-ink transition-transform hover:scale-[1.03]"
                >
                  <Github size={15} /> Clone the repository
                </a>
                <a
                  href="https://doi.org/10.1007/s44257-025-00054-1"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 rounded-full border border-white/20 px-7 py-3.5 text-xs font-bold uppercase tracking-wider text-bone transition-colors hover:border-gold hover:text-gold"
                >
                  <ExternalLink size={15} /> Discover Analytics paper
                </a>
              </div>
            </div>
            <div className="flex gap-16">
              <div>
                <span className="font-mono text-[10px] uppercase tracking-[0.25em] text-white/35">Authors</span>
                <p className="mt-3 text-sm text-white/70">Ricardo M. Leal Lopez</p>
              </div>
              <div>
                <span className="font-mono text-[10px] uppercase tracking-[0.25em] text-white/35">License</span>
                <p className="mt-3 text-sm text-white/70">CC-BY 4.0</p>
              </div>
            </div>
          </div>

          <div className="mt-16 flex flex-col gap-3 border-t border-white/10 pt-8 text-xs text-white/35 md:flex-row md:items-center md:justify-between">
            <span className="display text-base text-white/60">SDAO — Stochastic Diffusion Adaptive Optimization</span>
            <span>&copy; 2026 The Author(s). Licensed under Creative Commons Attribution 4.0 International (CC-BY 4.0).</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
