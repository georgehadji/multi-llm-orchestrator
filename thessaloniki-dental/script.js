/* 
 * Awwwards-Level Script for Thessaloniki Dental Excellence
 * Includes Custom Cursor, GSAP Scroll Animations, and Three.js WebGL Background.
 */

document.addEventListener("DOMContentLoaded", () => {
    
    // --- 1. Custom Cursor ---
    const cursor = document.querySelector('.cursor');
    const follower = document.querySelector('.cursor-follower');
    
    let mouseX = 0, mouseY = 0;
    let followerX = 0, followerY = 0;
    
    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
        
        // Instant cursor update
        gsap.to(cursor, {
            x: mouseX,
            y: mouseY,
            duration: 0.1
        });
    });
    
    // Follower easing animation
    gsap.ticker.add(() => {
        followerX += (mouseX - followerX) * 0.1;
        followerY += (mouseY - followerY) * 0.1;
        
        gsap.set(follower, {
            x: followerX,
            y: followerY
        });
    });

    // --- 2. GSAP Scroll Animations ---
    gsap.registerPlugin(ScrollTrigger);

    // Hero elements fade up
    gsap.from(".fade-up", {
        y: 50,
        opacity: 0,
        duration: 1,
        stagger: 0.2,
        ease: "power3.out",
        delay: 0.5
    });

    // Parallax effect for about section
    gsap.to(".about-image", {
        yPercent: -20,
        ease: "none",
        scrollTrigger: {
            trigger: ".about",
            start: "top bottom",
            end: "bottom top",
            scrub: true
        }
    });

    // Service cards stagger
    gsap.from(".service-card", {
        y: 100,
        opacity: 0,
        duration: 0.8,
        stagger: 0.2,
        scrollTrigger: {
            trigger: ".services",
            start: "top 80%"
        }
    });

    // --- 3. Three.js WebGL Background ---
    const container = document.getElementById('webgl-container');
    const scene = new THREE.Scene();
    
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 5;
    
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // Create a high-tech geometric shape (Icosahedron)
    const geometry = new THREE.IcosahedronGeometry(2, 1);
    
    // Material 1: Wireframe
    const materialWireframe = new THREE.MeshBasicMaterial({ 
        color: 0xd4af37, // Gold accent
        wireframe: true,
        transparent: true,
        opacity: 0.15
    });
    
    // Material 2: Points (Vertices)
    const materialPoints = new THREE.PointsMaterial({
        color: 0xffffff,
        size: 0.05,
        transparent: true,
        opacity: 0.8
    });

    const meshWireframe = new THREE.Mesh(geometry, materialWireframe);
    const meshPoints = new THREE.Points(geometry, materialPoints);
    
    // Group them
    const shapeGroup = new THREE.Group();
    shapeGroup.add(meshWireframe);
    shapeGroup.add(meshPoints);
    
    // Position it to the right
    shapeGroup.position.x = 2;
    scene.add(shapeGroup);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    // Mouse interaction for WebGL
    let targetX = 0;
    let targetY = 0;
    const windowHalfX = window.innerWidth / 2;
    const windowHalfY = window.innerHeight / 2;

    document.addEventListener('mousemove', (e) => {
        targetX = (e.clientX - windowHalfX) * 0.001;
        targetY = (e.clientY - windowHalfY) * 0.001;
    });

    // Scroll interaction for WebGL
    let scrollY = window.scrollY;
    window.addEventListener('scroll', () => {
        scrollY = window.scrollY;
    });

    // Animation Loop
    const clock = new THREE.Clock();

    function animate() {
        requestAnimationFrame(animate);
        const elapsedTime = clock.getElapsedTime();

        // Base rotation
        shapeGroup.rotation.y += 0.002;
        shapeGroup.rotation.x += 0.001;

        // Mouse interaction rotation
        shapeGroup.rotation.y += 0.05 * (targetX - shapeGroup.rotation.y);
        shapeGroup.rotation.x += 0.05 * (targetY - shapeGroup.rotation.x);

        // Scroll interaction (move object up as we scroll down)
        shapeGroup.position.y = -scrollY * 0.002;
        
        // Gentle breathing scale
        const scale = 1 + Math.sin(elapsedTime * 2) * 0.05;
        shapeGroup.scale.set(scale, scale, scale);

        renderer.render(scene, camera);
    }

    animate();

    // Window Resize Handler
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });

    // Form Submit Handler
    const form = document.getElementById('booking-form');
    if(form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            const btn = form.querySelector('.submit-btn');
            const originalText = btn.innerText;
            btn.innerText = "Processing...";
            
            setTimeout(() => {
                btn.innerText = "Request Sent";
                btn.style.color = "#d4af37";
                btn.style.borderColor = "#d4af37";
                setTimeout(() => {
                    form.reset();
                    btn.innerText = originalText;
                    btn.style.color = "";
                    btn.style.borderColor = "";
                }, 3000);
            }, 1500);
        });
    }
});
