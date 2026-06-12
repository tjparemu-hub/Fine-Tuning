// Intersection Observer — fade sections in on scroll
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.12 }
);

document.querySelectorAll(
  '.section__label, .section__heading, .about__bio, .about__divider, ' +
  '.service-card, .booking__card, .social-card, .booking__sub'
).forEach((el) => {
  el.classList.add('fade-in');
  observer.observe(el);
});

// Stagger service cards
document.querySelectorAll('.service-card').forEach((card, i) => {
  card.style.transitionDelay = `${i * 0.1}s`;
});

// Stagger social cards
document.querySelectorAll('.social-card').forEach((card, i) => {
  card.style.transitionDelay = `${i * 0.08}s`;
});
