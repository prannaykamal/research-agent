// Example research questions offered when a topic is refused. Pure, so it can be tested in Node.
// These are the distinct questions the evaluation runs used, so each is known to research well.

export const SAMPLE_TOPICS = [
  "How will lunar exploration and permanent Moon bases evolve over the next decade?",
  "How does generative AI affect software engineering productivity, code quality, and developer learning?",
  "How is artificial intelligence transforming early disease detection, medical diagnosis, and personalized treatment?",
  "How has hypersonic missile technology evolved from early research programs to modern systems?",
  "How can machine learning use EEG signals for reliable real-time stress detection?",
  "How have AI methods for drug discovery evolved from traditional machine learning to foundation models?",
  "How has organic chemistry evolved over time?",
  "How has climate change progressed since the Industrial Revolution, and what has driven it?",
  "How has the study of human behavior and cognition evolved from early psychology to modern neuroscience?",
  "How have cryogenic rocket engines evolved from the 1960s to the present?",
  "How has artificial intelligence evolved in healthcare, and what are its benefits and challenges?",
  "How has nuclear fusion research evolved, and what challenges remain before commercial fusion power?",
  "How have quantum computers evolved from theoretical concepts to current experimental systems?",
  "How have CRISPR-based technologies transformed modern genetic engineering?",
  "How has vaccine technology evolved from traditional approaches to mRNA vaccines?",
  "How has warfare evolved from conventional armies to cyber and autonomous systems?",
  "How has robotics evolved from industrial automation to autonomous humanoid systems?",
  "How have cryptocurrencies and blockchain changed the concept of digital money?",
  "How has internet infrastructure evolved from ARPANET to today's cloud and edge networks?",
  "How will small modular reactors shape nuclear power over the next decade?",
  "How have the cost and efficiency of solar photovoltaic panels changed since 2000?",
  "How has cancer treatment evolved from surgery and chemotherapy to targeted therapy and immunotherapy?",
  "How has solid-state battery technology developed, and what obstacles remain before mass-market EV adoption?",
  "How has antibiotic resistance developed since penicillin, and what new strategies are being used to combat it?",
  "How will autonomous vehicles change urban transportation over the next decade?",
  "How has wind turbine technology changed since 1990, and what limits further growth in turbine size?",
];

// `count` distinct questions in random order; `exclude` keeps a reshuffle from repeating the last set.
export function pickTopics(count = 3, { random = Math.random, exclude = [] } = {}) {
  const skip = new Set(exclude);
  const pool = SAMPLE_TOPICS.filter((topic) => !skip.has(topic));
  const source = pool.length >= count ? pool : [...SAMPLE_TOPICS];
  for (let index = source.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1));
    [source[index], source[other]] = [source[other], source[index]];
  }
  return source.slice(0, count);
}
