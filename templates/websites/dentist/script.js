/* Progressive enhancement only. The page is complete without this file:
   every .reveal element is visible by default and this merely animates it in.
   No dependencies, no inline script, nothing a CSP has to allow beyond 'self'. */
(function () {
  "use strict";

  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduced || !("IntersectionObserver" in window)) return;

  var targets = document.querySelectorAll("section");
  if (!targets.length) return;

  Array.prototype.forEach.call(targets, function (el) { el.classList.add("reveal"); });

  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("is-visible");
      observer.unobserve(entry.target);
    });
  }, { rootMargin: "0px 0px -10% 0px", threshold: 0.05 });

  Array.prototype.forEach.call(targets, function (el) { observer.observe(el); });
})();
