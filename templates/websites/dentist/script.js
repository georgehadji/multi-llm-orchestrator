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

// Form messaging. The markup carries the words (data-error / data-success) so a
// site in another language needs no JavaScript change, and the page still
// submits normally with scripting off — the browser's own validation takes over.
(function () {
  var form = document.querySelector('form.booking');
  if (!form) return;

  function messageFor(field) {
    var slot = field.getAttribute('aria-describedby');
    return slot ? document.getElementById(slot) : null;
  }

  function show(field) {
    var slot = messageFor(field);
    if (!slot) return;
    var invalid = !field.checkValidity();
    slot.textContent = invalid ? slot.getAttribute('data-error') || '' : '';
    field.setAttribute('aria-invalid', invalid ? 'true' : 'false');
  }

  Array.prototype.forEach.call(form.elements, function (field) {
    if (!field.name || field.type === 'submit') return;
    field.addEventListener('blur', function () { show(field); });
    field.addEventListener('input', function () {
      if (field.getAttribute('aria-invalid') === 'true') show(field);
    });
  });

  form.addEventListener('submit', function (event) {
    var status = form.querySelector('.form-status');
    if (!form.checkValidity()) {
      event.preventDefault();
      Array.prototype.forEach.call(form.elements, function (f) { if (f.name) show(f); });
      var first = form.querySelector('[aria-invalid="true"]');
      if (first) first.focus();
      return;
    }
    // The handler redirects on success; this covers the interim state so the
    // page is never silent after a click.
    if (status) status.textContent = status.getAttribute('data-success') || '';
  });
})();
