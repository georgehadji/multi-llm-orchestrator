/* 
 * script.js — Interactive rituals, booking assistant, and sliders.
 * Crafted with absolute precision.
 */

document.addEventListener('DOMContentLoaded', () => {

  // 1. Sticky Navigation on Scroll
  const navbar = document.getElementById('navbar');
  window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  });

  // 2. Mobile Menu Toggle
  const navToggle = document.getElementById('nav-toggle');
  const navLinks = document.getElementById('nav-links');
  
  if (navToggle && navLinks) {
    navToggle.addEventListener('click', () => {
      // Toggle active states
      navLinks.style.display = navLinks.style.display === 'flex' ? 'none' : 'flex';
      navToggle.classList.toggle('active');
    });

    // Close menu when a link is clicked
    document.querySelectorAll('.nav-item').forEach(link => {
      link.addEventListener('click', () => {
        if (window.innerWidth <= 768) {
          navLinks.style.display = 'none';
          navToggle.classList.remove('active');
        }
      });
    });
  }

  // 3. Services & Rituals Tabs Filter
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      const targetTab = button.getAttribute('data-tab');

      // Remove active from all buttons & panes
      tabButtons.forEach(btn => btn.classList.remove('active'));
      tabPanes.forEach(pane => pane.classList.remove('active'));

      // Add active to selected button & pane
      button.classList.add('active');
      const activePane = document.getElementById(targetTab);
      if (activePane) {
        activePane.classList.add('active');
      }
    });
  });

  // 4. Interactive Booking Assistant / Form Submission
  const appointmentForm = document.getElementById('appointment-form');
  const successCard = document.getElementById('booking-success-message');
  const bookAnotherBtn = document.getElementById('book-another-btn');

  // Success fields
  const successClientName = document.getElementById('success-client-name');
  const successRitual = document.getElementById('success-ritual');
  const successBarber = document.getElementById('success-barber');
  const successDate = document.getElementById('success-date');
  const successTime = document.getElementById('success-time');

  if (appointmentForm && successCard) {
    appointmentForm.addEventListener('submit', (e) => {
      e.preventDefault();

      // Retrieve form values
      const name = document.getElementById('client-name').value;
      const ritualSelect = document.getElementById('ritual-type');
      const ritualText = ritualSelect.options[ritualSelect.selectedIndex].text;
      const barberSelect = document.getElementById('barber-preference');
      const barberText = barberSelect.options[barberSelect.selectedIndex].text;
      const date = document.getElementById('appointment-date').value;
      const time = document.getElementById('appointment-time').value;

      // Populate success card
      successClientName.textContent = name;
      successRitual.textContent = ritualText;
      successBarber.textContent = barberText;
      successDate.textContent = date;
      successTime.textContent = time;

      // Toggle display
      appointmentForm.classList.add('hidden');
      successCard.classList.remove('hidden');

      // Smooth scroll to success card
      successCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });

    if (bookAnotherBtn) {
      bookAnotherBtn.addEventListener('click', () => {
        // Reset form
        appointmentForm.reset();
        
        // Toggle visibility back
        successCard.classList.add('hidden');
        appointmentForm.classList.remove('hidden');
      });
    }
  }

  // Set minimum date of calendar to today's date
  const dateInput = document.getElementById('appointment-date');
  if (dateInput) {
    const today = new Date().toISOString().split('T')[0];
    dateInput.min = today;
  }

  // 5. Sliding Testimonials Carousel
  const slides = document.querySelectorAll('.testimonial-slide');
  const dots = document.querySelectorAll('.slider-dot');
  let currentSlide = 0;
  let slideInterval;

  function showSlide(index) {
    slides.forEach(slide => slide.classList.remove('active'));
    dots.forEach(dot => dot.classList.remove('active'));

    slides[index].classList.add('active');
    dots[index].classList.add('active');
    currentSlide = index;
  }

  function nextSlide() {
    let next = currentSlide + 1;
    if (next >= slides.length) {
      next = 0;
    }
    showSlide(next);
  }

  // Manual dot navigation
  dots.forEach(dot => {
    dot.addEventListener('click', () => {
      clearInterval(slideInterval); // Stop auto-play upon manual interaction
      const targetIndex = parseInt(dot.getAttribute('data-index'));
      showSlide(targetIndex);
      startAutoPlay(); // Restart auto-play
    });
  });

  function startAutoPlay() {
    slideInterval = setInterval(nextSlide, 6000); // Transition every 6 seconds
  }

  // Initiate Slider
  if (slides.length > 0) {
    showSlide(0);
    startAutoPlay();
  }

});
