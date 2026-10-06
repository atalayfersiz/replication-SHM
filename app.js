// Civil-DT Dashboard Logic
// Replicates structural metrics and ML comparisons for Sydney Footbridge (Hielscher et al., 2023)

let dtData = null;
let currentT = 0;
let isPlaying = false;
let playbackSpeed = 3; // Steps per frame
let activeArray = 'A'; // 'A' or 'B'
let activeModel = 'ANN'; // 'ANN', 'Linear', 'Ridge', 'Lasso'
let activeSensorPos = 14; // Default to 14.0m
let activeDensity = 'Full'; // 'Full', 'Half', 'Sparse'
let sessionAlertHistory = [];

function getActiveKey() {
    let suffix = '';
    if (activeDensity === 'Half') suffix = '_Half';
    else if (activeDensity === 'Sparse') suffix = '_Sparse';
    return `Array_${activeArray}${suffix}`;
}

// Three.js global variables
let scene, camera, renderer, controls;
let bridgeDeckMesh, columnSupportMesh, palletJackMesh, loadPointLight;
let sensorSpheres = [];
let deflectionRibbon;

// Chart.js and state global variables
let momentChart, deflectionChart, timeSeriesChart, stiffnessChart;
let alertActive = false;
let wasAlertActive = false;

// HTML Elements
const modelSelect = document.getElementById('model-select');
const btnArrA = document.getElementById('btn-arr-a');
const btnArrB = document.getElementById('btn-arr-b');
const btnPlay = document.getElementById('btn-play');
const playIcon = btnPlay.querySelector('.play-icon');
const pauseIcon = btnPlay.querySelector('.pause-icon');
const timelineSlider = document.getElementById('timeline-slider');
const speedSlider = document.getElementById('speed-slider');
const timeDisplay = document.getElementById('time-display');
const posDisplay = document.getElementById('pos-display');
const sensorSelect = document.getElementById('sensor-select');
const btnResetCam = document.getElementById('btn-reset-cam');
const validationModal = document.getElementById('validation-modal');
const btnValidationInfo = document.getElementById('btn-validation-info');
const btnCloseModal = document.getElementById('btn-close-modal');
const alertsModal = document.getElementById('alerts-modal');
const btnExpandAlerts = document.getElementById('btn-expand-alerts');
const btnCloseAlertsModal = document.getElementById('btn-close-alerts-modal');
const modalAlertsContainer = document.getElementById('modal-alerts-container');

// KPI Elements
const kpiLoadPos = document.getElementById('kpi-load-pos');
const kpiMaxStrain = document.getElementById('kpi-max-strain');
const kpiMaxMoment = document.getElementById('kpi-max-moment');
const kpiMaxDeflection = document.getElementById('kpi-max-deflection');

// Initialize Lucide Icons
lucide.createIcons();

// Load the JSON data
fetch('backend/frontend_data.json')
    .then(response => {
        if (!response.ok) {
            throw new Error("HTTP error " + response.status);
        }
        return response.json();
    })
    .then(data => {
        dtData = data;
        console.log("Loaded frontend data successfully", dtData);
        
        // Initialize everything
        initCharts();
        initThree();
        setupEventListeners();
        animate();
        updateDashboard();
    })
    .catch(error => {
        console.error("Failed to load frontend data:", error);
        alert("Could not load backend/frontend_data.json. Please run 'python backend/structural_analysis.py' first and serve via an HTTP server.");
    });

// Setup Event Listeners
function setupEventListeners() {
    // Play/Pause
    btnPlay.addEventListener('click', () => {
        isPlaying = !isPlaying;
        if (isPlaying) {
            playIcon.classList.add('hidden');
            pauseIcon.classList.remove('hidden');
        } else {
            playIcon.classList.remove('hidden');
            pauseIcon.classList.add('hidden');
        }
    });

    // Timeline Slider
    timelineSlider.addEventListener('input', (e) => {
        currentT = parseInt(e.target.value);
        updateDashboard();
    });

    // Speed Slider
    speedSlider.addEventListener('input', (e) => {
        playbackSpeed = parseInt(e.target.value);
    });

    // Model Selector
    modelSelect.addEventListener('change', (e) => {
        activeModel = e.target.value;
        // Update highlight row in table
        document.querySelectorAll('.metrics-table tr').forEach(tr => tr.classList.remove('highlight-row'));
        document.getElementById(`row-${activeModel.toLowerCase()}`).classList.add('highlight-row');
        updateChartsStaticData();
        updateDashboard();
    });

    // Sensor Selector
    sensorSelect.addEventListener('change', (e) => {
        activeSensorPos = parseInt(e.target.value);
        updateTimeSeriesChart();
    });

    // Reset Camera
    btnResetCam.addEventListener('click', () => {
        if (controls) {
            controls.reset();
            camera.position.set(9.0, 4.0, 10.0);
            controls.target.set(8.95, 0, 0);
            controls.update();
        }
    });

    // Sensor Density Selector
    const densitySelect = document.getElementById('density-select');
    densitySelect.addEventListener('change', (e) => {
        activeDensity = e.target.value;
        updateChartsStaticData();
        updateDashboard();
    });

    // Open Validation Modal
    btnValidationInfo.addEventListener('click', () => {
        validationModal.classList.remove('hidden');
    });

    // Close Validation Modal (via button)
    btnCloseModal.addEventListener('click', () => {
        validationModal.classList.add('hidden');
    });

    // Close Validation Modal (via clicking overlay backdrop)
    validationModal.addEventListener('click', (e) => {
        if (e.target === validationModal) {
            validationModal.classList.add('hidden');
        }
    });

    // Open Alerts Modal
    btnExpandAlerts.addEventListener('click', () => {
        alertsModal.classList.remove('hidden');
        updateModalAlertsView();
    });

    // Close Alerts Modal
    btnCloseAlertsModal.addEventListener('click', () => {
        alertsModal.classList.add('hidden');
    });

    alertsModal.addEventListener('click', (e) => {
        if (e.target === alertsModal) {
            alertsModal.classList.add('hidden');
        }
    });
}

function setArray(arr) {
    activeArray = arr;
    if (arr === 'A') {
        btnArrA.classList.add('active');
        btnArrB.classList.remove('active');
    } else {
        btnArrB.classList.add('active');
        btnArrA.classList.remove('active');
    }
    updateChartsStaticData();
    updateDashboard();
}

// ----------------------------------------------------
// THREE.JS BRIDGE VISUALIZATION
// ----------------------------------------------------
function initThree() {
    const container = document.getElementById('three-canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Create Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0912);
    scene.fog = new THREE.FogExp2(0x0a0912, 0.035);

    // Create Camera
    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(9.0, 4.0, 10.0);

    // Create Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // Add OrbitControls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 - 0.05; // Don't go below ground
    controls.minDistance = 3;
    controls.maxDistance = 25;
    controls.target.set(8.95, 0, 0);
    controls.update();

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.2);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.7);
    dirLight.position.set(9, 15, 5);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 1024;
    dirLight.shadow.mapSize.height = 1024;
    dirLight.shadow.bias = -0.001;
    scene.add(dirLight);

    // Subtle blueish fill light
    const fillLight = new THREE.DirectionalLight(0x3b82f6, 0.4);
    fillLight.position.set(-5, 5, -5);
    scene.add(fillLight);

    // Ground plane
    const groundGeo = new THREE.PlaneGeometry(60, 40);
    const groundMat = new THREE.MeshStandardMaterial({
        color: 0x08070d,
        roughness: 0.9,
        metalness: 0.1
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -3.5;
    ground.receiveShadow = true;
    scene.add(ground);

    // Grid helper representing riverbed / ground
    const gridHelper = new THREE.GridHelper(50, 50, 0x1f1d35, 0x11101e);
    gridHelper.position.y = -3.49;
    scene.add(gridHelper);

    // BRIDGE GEOMETRY
    // Concrete Deck
    const deckLength = 17.9;
    const deckWidth = 2.6;
    const deckDepth = 0.28;
    
    // We create a fine-segmented box to allow showing deflection in the bridge deck geometry later if wanted.
    // Deck center is at x=8.95, y=-deckDepth/2, z=0
    const deckGeo = new THREE.BoxGeometry(deckLength, deckDepth, deckWidth, 36, 1, 4);
    const deckMat = new THREE.MeshStandardMaterial({
        color: 0x333147,
        roughness: 0.7,
        metalness: 0.2,
        flatShading: false
    });
    bridgeDeckMesh = new THREE.Mesh(deckGeo, deckMat);
    bridgeDeckMesh.position.set(8.95, -deckDepth/2, 0);
    bridgeDeckMesh.castShadow = true;
    bridgeDeckMesh.receiveShadow = true;
    scene.add(bridgeDeckMesh);

    // Support piers
    // Pinned ends
    const supportGeo = new THREE.BoxGeometry(0.6, 0.6, 2.8);
    const supportMat = new THREE.MeshStandardMaterial({ color: 0x1f1b2e, roughness: 0.8 });
    
    const supportLeft = new THREE.Mesh(supportGeo, supportMat);
    supportLeft.position.set(0, -0.3 - deckDepth, 0);
    supportLeft.castShadow = true;
    scene.add(supportLeft);

    const supportRight = new THREE.Mesh(supportGeo, supportMat);
    supportRight.position.set(deckLength, -0.3 - deckDepth, 0);
    supportRight.castShadow = true;
    scene.add(supportRight);

    // Column pier at 9.0m
    const pierHeight = 3.5 - deckDepth;
    const pierGeo = new THREE.CylinderGeometry(0.35, 0.45, pierHeight, 16);
    const pierMat = new THREE.MeshStandardMaterial({
        color: 0x27243c,
        roughness: 0.6
    });
    columnSupportMesh = new THREE.Mesh(pierGeo, pierMat);
    // Cylinder is centered around its height
    columnSupportMesh.position.set(9.0, -pierHeight/2 - deckDepth, 0);
    columnSupportMesh.castShadow = true;
    columnSupportMesh.receiveShadow = true;
    scene.add(columnSupportMesh);

    // Foundation block for pier
    const foundGeo = new THREE.BoxGeometry(1.5, 0.4, 1.5);
    const foundMesh = new THREE.Mesh(foundGeo, supportMat);
    foundMesh.position.set(9.0, -3.3, 0);
    foundMesh.castShadow = true;
    scene.add(foundMesh);

    // Sleek metallic handrails
    const railMat = new THREE.MeshStandardMaterial({ color: 0x8b5cf6, metalness: 0.8, roughness: 0.2 });
    const railGeo = new THREE.BoxGeometry(deckLength, 0.05, 0.05);
    
    const railLeft = new THREE.Mesh(railGeo, railMat);
    railLeft.position.set(8.95, 0.8, -deckWidth/2 + 0.05);
    scene.add(railLeft);

    const railRight = new THREE.Mesh(railGeo, railMat);
    railRight.position.set(8.95, 0.8, deckWidth/2 - 0.05);
    scene.add(railRight);

    // Posts for handrails
    const postGeo = new THREE.CylinderGeometry(0.02, 0.02, 0.8, 8);
    for (let x = 0.5; x < deckLength; x += 1.5) {
        const postL = new THREE.Mesh(postGeo, railMat);
        postL.position.set(x, 0.4, -deckWidth/2 + 0.05);
        scene.add(postL);

        const postR = new THREE.Mesh(postGeo, railMat);
        postR.position.set(x, 0.4, deckWidth/2 - 0.05);
        scene.add(postR);
    }

    // Moving Load (Pallet Jack / Weight)
    const loadGeo = new THREE.BoxGeometry(0.6, 0.4, 0.5);
    const loadMat = new THREE.MeshStandardMaterial({
        color: 0xec4899,
        emissive: 0x4a0e2e,
        roughness: 0.3,
        metalness: 0.8
    });
    palletJackMesh = new THREE.Mesh(loadGeo, loadMat);
    palletJackMesh.position.set(2.0, 0.2, 0); // moves along x, y center is 0.2
    palletJackMesh.castShadow = true;
    scene.add(palletJackMesh);

    // Point light on the load
    loadPointLight = new THREE.PointLight(0x06b6d4, 2, 4);
    loadPointLight.position.set(0, 0.6, 0);
    loadPointLight.castShadow = true;
    loadPointLight.shadow.bias = -0.002;
    palletJackMesh.add(loadPointLight);

    // Small glowing wheels for the load
    const wheelGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.1, 12);
    wheelGeo.rotateZ(Math.PI/2);
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x11101e, roughness: 0.5 });
    
    const wheel1 = new THREE.Mesh(wheelGeo, wheelMat); wheel1.position.set(-0.2, -0.2, -0.2); palletJackMesh.add(wheel1);
    const wheel2 = new THREE.Mesh(wheelGeo, wheelMat); wheel2.position.set(0.2, -0.2, -0.2); palletJackMesh.add(wheel2);
    const wheel3 = new THREE.Mesh(wheelGeo, wheelMat); wheel3.position.set(-0.2, -0.2, 0.2); palletJackMesh.add(wheel3);
    const wheel4 = new THREE.Mesh(wheelGeo, wheelMat); wheel4.position.set(0.2, -0.2, 0.2); palletJackMesh.add(wheel4);

    // Sensor nodes (3D spheres)
    const sphereGeo = new THREE.SphereGeometry(0.06, 16, 16);
    sensorSpheres = [];

    // Create 16 locations, Top and Bot, Array A and B (total 64)
    dtData.x_sensors.forEach((xVal) => {
        // Array A (z = -0.8)
        const nodeATop = new THREE.Mesh(sphereGeo, new THREE.MeshBasicMaterial({ color: 0x06b6d4 }));
        nodeATop.position.set(xVal, 0.05, -0.8);
        scene.add(nodeATop);

        const nodeABot = new THREE.Mesh(sphereGeo, new THREE.MeshBasicMaterial({ color: 0x06b6d4 }));
        nodeABot.position.set(xVal, -deckDepth - 0.05, -0.8);
        scene.add(nodeABot);

        // Array B (z = 0.8)
        const nodeBTop = new THREE.Mesh(sphereGeo, new THREE.MeshBasicMaterial({ color: 0x06b6d4 }));
        nodeBTop.position.set(xVal, 0.05, 0.8);
        scene.add(nodeBTop);

        const nodeBBot = new THREE.Mesh(sphereGeo, new THREE.MeshBasicMaterial({ color: 0x06b6d4 }));
        nodeBBot.position.set(xVal, -deckDepth - 0.05, 0.8);
        scene.add(nodeBBot);

        sensorSpheres.push({
            x: xVal,
            A: { top: nodeATop, bot: nodeABot },
            B: { top: nodeBTop, bot: nodeBBot }
        });
    });

    // Deflected line indicator ribbon (glowing tubes running underneath the active array deck)
    const ribbonGeo = new THREE.CylinderGeometry(0.025, 0.025, 1, 8);
    ribbonGeo.rotateZ(Math.PI/2);
    const ribbonMat = new THREE.MeshBasicMaterial({ color: 0x00f5d4 });
    
    // We create a line group or segment mesh
    deflectionRibbon = new THREE.Group();
    scene.add(deflectionRibbon);

    // Handle resize
    window.addEventListener('resize', onWindowResize, false);
}

function onWindowResize() {
    const container = document.getElementById('three-canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

// ----------------------------------------------------
// CHART.JS INITIALIZATION
// ----------------------------------------------------
function initCharts() {
    const xLabels = dtData.x_sensors.map(x => `${x.toFixed(1)}m`);

    // Bending Moment Chart (Chart 1)
    const ctxMoment = document.getElementById('moment-chart').getContext('2d');
    momentChart = new Chart(ctxMoment, {
        type: 'line',
        data: {
            labels: xLabels,
            datasets: [
                {
                    label: 'Calculated (DT)',
                    data: new Array(16).fill(0),
                    borderColor: 'rgb(6, 182, 212)',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    borderWidth: 3,
                    pointRadius: 4,
                    pointBackgroundColor: 'rgb(6, 182, 212)',
                    tension: 0.3
                },
                {
                    label: 'Paper DT',
                    data: dtData.Paper_DT_A.bending_moments,
                    borderColor: 'rgba(236, 72, 153, 0.45)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3
                },
                {
                    label: 'Paper FE',
                    data: dtData.Paper_FE_A.bending_moments,
                    borderColor: 'rgba(139, 92, 246, 0.45)',
                    borderWidth: 2,
                    borderDash: [2, 2],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9CA3AF' },
                    title: { display: true, text: 'Moment (kNm)', color: '#9CA3AF' }
                },
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9CA3AF' }
                }
            }
        }
    });

    // Deflection Chart (Chart 2)
    const ctxDeflection = document.getElementById('deflection-chart').getContext('2d');
    deflectionChart = new Chart(ctxDeflection, {
        type: 'line',
        data: {
            labels: xLabels,
            datasets: [
                {
                    label: 'Calculated (DT)',
                    data: new Array(16).fill(0),
                    borderColor: 'rgb(6, 182, 212)',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    borderWidth: 3,
                    pointRadius: 4,
                    pointBackgroundColor: 'rgb(6, 182, 212)',
                    tension: 0.3
                },
                {
                    label: 'Paper DT',
                    data: dtData.Paper_DT_A.deflections,
                    borderColor: 'rgba(236, 72, 153, 0.45)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3
                },
                {
                    label: 'Paper FE',
                    data: dtData.Paper_FE_A.deflections,
                    borderColor: 'rgba(139, 92, 246, 0.45)',
                    borderWidth: 2,
                    borderDash: [2, 2],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9CA3AF' },
                    title: { display: true, text: 'Deflection (mm)', color: '#9CA3AF' }
                },
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9CA3AF' }
                }
            }
        }
    });

    // Concrete Stiffness Retention Chart (Chart 3)
    const ctxStiffness = document.getElementById('stiffness-chart').getContext('2d');
    stiffnessChart = new Chart(ctxStiffness, {
        type: 'line',
        data: {
            labels: xLabels,
            datasets: [
                {
                    label: 'Stiffness Retention (%)',
                    data: new Array(16).fill(100),
                    borderColor: 'rgb(239, 68, 68)',
                    backgroundColor: 'rgba(239, 68, 68, 0.05)',
                    borderWidth: 2,
                    pointRadius: 4,
                    pointBackgroundColor: 'rgb(239, 68, 68)',
                    fill: true,
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    min: 0,
                    max: 105,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9CA3AF' },
                    title: { display: true, text: 'Stiffness Retention (%)', color: '#9CA3AF' }
                },
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9CA3AF' }
                }
            }
        }
    });

    // Time Series Strains & Temperatures (Chart 4)
    const ctxTimeSeries = document.getElementById('time-series-chart').getContext('2d');
    
    // We generate time array for X-axis labels
    const tLabels = dtData.timestamps.map(t => {
        const m = Math.floor(t / 60);
        const s = t % 60;
        return `${m}:${s.toString().padStart(2, '0')}`;
    });

    timeSeriesChart = new Chart(ctxTimeSeries, {
        type: 'line',
        data: {
            labels: tLabels,
            datasets: [
                {
                    label: 'Raw Strain Bot (με)',
                    data: [],
                    borderColor: 'rgb(244, 63, 94)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false,
                    yAxisID: 'y'
                },
                {
                    label: 'Baseline Pred Bot (με)',
                    data: [],
                    borderColor: 'rgba(244, 63, 94, 0.4)',
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    fill: false,
                    yAxisID: 'y'
                },
                {
                    label: 'Raw Strain Top (με)',
                    data: [],
                    borderColor: 'rgb(59, 130, 246)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false,
                    yAxisID: 'y'
                },
                {
                    label: 'Baseline Pred Top (με)',
                    data: [],
                    borderColor: 'rgba(59, 130, 246, 0.4)',
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    fill: false,
                    yAxisID: 'y'
                },
                {
                    label: 'Temperature (°C)',
                    data: [],
                    borderColor: 'rgb(245, 158, 11)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false,
                    yAxisID: 'yTemp'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#9CA3AF', boxWidth: 15, font: { size: 10 } }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            scales: {
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#9CA3AF' },
                    title: { display: true, text: 'Strain (με)', color: '#9CA3AF' }
                },
                yTemp: {
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#9CA3AF' },
                    title: { display: true, text: 'Temperature (°C)', color: '#9CA3AF' }
                },
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#9CA3AF', maxTicksLimit: 16 }
                }
            }
        },
        // Draw vertical cursor at current simulation time
        plugins: [{
            id: 'current-time-cursor',
            afterDraw: (chart) => {
                const activeIndex = currentT;
                if (activeIndex >= 0 && activeIndex < chart.scales.x.ticks.length) {
                    const ctx = chart.ctx;
                    const x = chart.scales.x.getPixelForValue(activeIndex);
                    const topY = chart.scales.y.top;
                    const bottomY = chart.scales.y.bottom;

                    ctx.save();
                    ctx.beginPath();
                    ctx.moveTo(x, topY);
                    ctx.lineTo(x, bottomY);
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = '#00f5d4';
                    ctx.shadowColor = '#00f5d4';
                    ctx.shadowBlur = 8;
                    ctx.stroke();
                    ctx.restore();
                }
            }
        }]
    });

    // Initial update of time series static arrays
    updateTimeSeriesChart();
}

// Interplates local temperature from the 4 temperature sensors at a given span coordinate x
function getLocalTemperatureSeries(x) {
    const t02 = dtData.temperatures.TMP_02m;
    const t07 = dtData.temperatures.TMP_07m;
    const t12 = dtData.temperatures.TMP_12m;
    const t17 = dtData.temperatures.TMP_17m;
    
    const count = t02.length;
    const result = new Array(count);
    
    for (let t = 0; t < count; t++) {
        if (x <= 7.0) {
            let frac = (x - 2.0) / 5.0;
            result[t] = t02[t] * (1.0 - frac) + t07[t] * frac;
        } else if (x <= 12.0) {
            let frac = (x - 7.0) / 5.0;
            result[t] = t07[t] * (1.0 - frac) + t12[t] * frac;
        } else {
            let frac = (x - 12.0) / 5.0;
            result[t] = t12[t] * (1.0 - frac) + t17[t] * frac;
        }
    }
    return result;
}

// Update static lines on charts when active model/array changes (e.g. Paper comparisons at load=14m)
function updateChartsStaticData() {
    const isA = (activeArray === 'A');
    
    // Choose correct paper data
    let momentsPaperDT, momentsPaperFE;
    let deflectionsPaperDT, deflectionsPaperFE;

    if (isA) {
        momentsPaperDT = dtData.Paper_DT_A.bending_moments;
        momentsPaperFE = dtData.Paper_FE_A.bending_moments;
        deflectionsPaperDT = dtData.Paper_DT_A.deflections;
        deflectionsPaperFE = dtData.Paper_FE_A.deflections;
    } else {
        momentsPaperDT = dtData.Paper_DT_B.bending_moments;
        momentsPaperFE = dtData.Paper_FE_B.bending_moments;
        deflectionsPaperDT = dtData.Paper_DT_B.deflections;
        deflectionsPaperFE = dtData.Paper_FE_B.deflections;
    }

    momentChart.data.datasets[1].data = momentsPaperDT;
    momentChart.data.datasets[2].data = momentsPaperFE;
    momentChart.update('none');

    deflectionChart.data.datasets[1].data = deflectionsPaperDT;
    deflectionChart.data.datasets[2].data = deflectionsPaperFE;
    deflectionChart.update('none');

    // Update time series chart
    updateTimeSeriesChart();
}

// Updates the time series chart dataset when selected sensor changes
function updateTimeSeriesChart() {
    if (!dtData) return;
    
    // Find index of the selected sensor coordinate
    const sensorIdx = dtData.x_sensors.indexOf(activeSensorPos);
    if (sensorIdx === -1) return;

    const key = getActiveKey();
    const dataset = dtData.models[activeModel][key];

    // Build data arrays for selected sensor position across all 960 seconds
    const rawBotData = dataset.map(item => item.raw_bot[sensorIdx]);
    const baseBotData = dataset.map(item => item.base_bot[sensorIdx]);
    const rawTopData = dataset.map(item => item.raw_top[sensorIdx]);
    const baseTopData = dataset.map(item => item.base_top[sensorIdx]);
    const tempSeries = getLocalTemperatureSeries(activeSensorPos);

    timeSeriesChart.data.datasets[0].data = rawBotData;
    timeSeriesChart.data.datasets[1].data = baseBotData;
    timeSeriesChart.data.datasets[2].data = rawTopData;
    timeSeriesChart.data.datasets[3].data = baseTopData;
    timeSeriesChart.data.datasets[4].data = tempSeries;
    
    timeSeriesChart.update('none');
}

// ----------------------------------------------------
// SIMULATION & RUNTIME UPDATE LOOP
// ----------------------------------------------------
function updateDashboard() {
    if (!dtData) return;

    const t = currentT;
    const isA = (activeArray === 'A');
    const key = getActiveKey();
    
    // Get simulation frames for active model & array configuration
    const simFrame = dtData.models[activeModel][key][t];
    const loadPos = dtData.load_positions[t];

    // 1. UPDATE TIMELINE & LABELS
    const minutes = Math.floor(t / 60);
    const seconds = t % 60;
    timeDisplay.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    posDisplay.textContent = `Load: ${loadPos.toFixed(2)}m`;
    timelineSlider.value = t;

    // 2. CALCULATE MAX METRICS FOR KPIs
    // Strains
    const deltaTopArr = simFrame.strains_top;
    const deltaBotArr = simFrame.strains_bot;
    let maxDelta = 0;
    for (let i = 0; i < 16; i++) {
        const absTop = Math.abs(deltaTopArr[i]);
        const absBot = Math.abs(deltaBotArr[i]);
        if (absTop > maxDelta) maxDelta = absTop;
        if (absBot > maxDelta) maxDelta = absBot;
    }

    // Moments
    const moments = simFrame.bending_moments;
    const maxMoment = Math.max(...moments.map(Math.abs));

    // Deflections
    const deflections = simFrame.deflections;
    const maxDeflection = Math.max(...deflections.map(Math.abs));

    // Update KPI panels
    kpiLoadPos.textContent = `${loadPos.toFixed(2)} m`;
    kpiMaxStrain.textContent = `${maxDelta.toFixed(1)} με`;
    kpiMaxMoment.textContent = `${maxMoment.toFixed(2)} kNm`;
    kpiMaxMoment.style.color = (maxMoment > 5.0) ? 'var(--danger)' : '';
    kpiMaxDeflection.textContent = `${maxDeflection.toFixed(2)} mm`;
    kpiMaxDeflection.style.color = (maxDeflection > 0.15) ? 'var(--danger)' : '';

    // 3. UPDATE CHART.JS GRAPHS
    // Live bending moment
    momentChart.data.datasets[0].data = moments;
    momentChart.update('none');

    // Live deflection
    deflectionChart.data.datasets[0].data = deflections;
    deflectionChart.update('none');

    // Live stiffness retention
    const stiffnessRatios = simFrame.stiffness_ratios;
    if (stiffnessChart) {
        stiffnessChart.data.datasets[0].data = stiffnessRatios;
        stiffnessChart.update('none');
    }

    // Redraw time series vertical cursor
    timeSeriesChart.update('none');

    // 4. REAL-TIME ALERT CHECKS
    let activeAlerts = [];
    alertActive = false;

    moments.forEach((m, idx) => {
        const sensorX = dtData.x_sensors[idx];
        if (Math.abs(m) > 5.0) {
            alertActive = true;
            activeAlerts.push(`Moment Exceeded: ${m.toFixed(2)} kNm at ${sensorX.toFixed(1)}m (Limit: 5.0 kNm)`);
        }
    });

    deflections.forEach((w, idx) => {
        const sensorX = dtData.x_sensors[idx];
        if (Math.abs(w) > 0.15) {
            alertActive = true;
            activeAlerts.push(`Deflection Exceeded: ${w.toFixed(2)} mm at ${sensorX.toFixed(1)}m (Limit: 0.15 mm)`);
        }
    });

    const statusIndicator = document.querySelector('.status-indicator');
    const statusText = statusIndicator.querySelector('.status-text');
    const alertLogContainer = document.getElementById('alert-log-container');
    const alertSection = document.querySelector('.alert-section');

    const timeStr = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;

    if (alertActive) {
        statusIndicator.classList.add('alert-active');
        statusText.textContent = 'ALERT: LIMIT EXCEEDED';
        if (alertSection) alertSection.classList.add('alert-active');
        
        // Play sound when transition to alert state occurs
        if (!wasAlertActive) {
            playAlertSound();
        }

        // Add new alerts to history
        activeAlerts.forEach(alert => {
            const formatted = `[t=${timeStr}] ${alert}`;
            if (!sessionAlertHistory.includes(formatted)) {
                sessionAlertHistory.push(formatted);
            }
        });

        alertLogContainer.innerHTML = activeAlerts.map(alert => `<div class="alert-item">[t=${timeStr}] ${alert}</div>`).join('');
    } else {
        statusIndicator.classList.remove('alert-active');
        statusText.textContent = 'OPERATIONAL';
        if (alertSection) alertSection.classList.remove('alert-active');
        alertLogContainer.innerHTML = '<div class="alert-empty-state">No structural alerts. System green.</div>';
    }

    wasAlertActive = alertActive;

    // 5. UPDATE THREE.JS MESHES
    // Update moving load position
    if (palletJackMesh) {
        palletJackMesh.position.x = loadPos;
    }

    // Color code and position 3D sensor nodes based on live strains & alert status
    const maxScaleStrain = 4.0;
    
    sensorSpheres.forEach((sensor, index) => {
        // Determine if this sensor is active under current density selection
        let isSensorActive = true;
        if (activeDensity === 'Half' && index % 2 !== 0) {
            isSensorActive = false;
        } else if (activeDensity === 'Sparse' && index % 4 !== 0) {
            isSensorActive = false;
        }

        const valA_top = dtData.models[activeModel]['Array_A'][t].strains_top[index];
        const valA_bot = dtData.models[activeModel]['Array_A'][t].strains_bot[index];
        const valB_top = dtData.models[activeModel]['Array_B'][t].strains_top[index];
        const valB_bot = dtData.models[activeModel]['Array_B'][t].strains_bot[index];

        // Check if this sensor location is in an alert state
        const isExceeded = (Math.abs(moments[index]) > 5.0) || (Math.abs(deflections[index]) > 0.15);

        const meshes = [
            { mesh: sensor.A.top, value: valA_top, array: 'A' },
            { mesh: sensor.A.bot, value: valA_bot, array: 'A' },
            { mesh: sensor.B.top, value: valB_top, array: 'B' },
            { mesh: sensor.B.bot, value: valB_bot, array: 'B' }
        ];

        meshes.forEach(item => {
            if (isSensorActive) {
                item.mesh.material.opacity = 1.0;
                item.mesh.material.transparent = false;
                
                if (isExceeded) {
                    // Flashing alert state (modulating color and size using sine wave)
                    const flashTime = Date.now() * 0.015;
                    const isFlashOn = Math.floor(flashTime) % 2 === 0;
                    if (isFlashOn) {
                        item.mesh.material.color.setHex(0xff0000); // Bright Red
                    } else {
                        colorSensor(item.mesh, item.value, maxScaleStrain);
                    }
                    const scaleBase = (item.array === activeArray) ? 1.6 : 1.3;
                    const pulse = scaleBase * (1.0 + 0.15 * Math.sin(flashTime));
                    item.mesh.scale.set(pulse, pulse, pulse);
                } else {
                    colorSensor(item.mesh, item.value, maxScaleStrain);
                    const scale = (item.array === activeArray) ? 1.3 : 1.0;
                    item.mesh.scale.set(scale, scale, scale);
                }
            } else {
                // Inactive sensor: grey, tiny and transparent to represent disabled state
                item.mesh.material.opacity = 0.15;
                item.mesh.material.transparent = true;
                item.mesh.material.color.setHex(0x555555); // Flat grey
                item.mesh.scale.set(0.4, 0.4, 0.4);
            }
        });
    });

    // Exaggerated Deflected Ribbon
    updateDeflectionRibbon(deflections);
}

function colorSensor(mesh, value, maxVal) {
    // Map value to HSL
    // Positive (tension) -> Red/Orange
    // Negative (compression) -> Blue
    // Near zero -> neutral grey/green
    
    let hue = 180; // Default cyan/neutral
    let saturation = 0;
    let lightness = 50;

    const clampVal = Math.min(Math.max(value, -maxVal), maxVal);
    const frac = Math.abs(clampVal) / maxVal;

    if (clampVal > 0.05) {
        hue = 350; // Rose red
        saturation = Math.round(frac * 100);
        lightness = 50 + Math.round((1 - frac) * 15);
    } else if (clampVal < -0.05) {
        hue = 220; // Royal blue
        saturation = Math.round(frac * 100);
        lightness = 50 + Math.round((1 - frac) * 15);
    } else {
        hue = 180;
        saturation = 10;
        lightness = 40;
    }

    mesh.material.color.setHSL(hue / 360.0, saturation / 100.0, lightness / 100.0);
}

function updateDeflectionRibbon(deflections) {
    if (!deflectionRibbon) return;

    // Clear old ribbon segments
    while (deflectionRibbon.children.length > 0) {
        const obj = deflectionRibbon.children[0];
        deflectionRibbon.remove(obj);
        obj.geometry.dispose();
    }

    // Set coordinate constants
    const zOffset = (activeArray === 'A') ? -0.8 : 0.8;
    const yBaseline = -0.28; // Bottom of deck
    
    // We construct 3D segments connecting the deflection points
    // Deflection is double integrated along 16 sensor locations [2.0m, ..., 17.0m]
    // Pinned ends have 0 deflection at x=0 and x=17.9
    
    const xVals = [0.0, ...dtData.x_sensors, 17.9];
    const dVals = [0.0, ...deflections, 0.0];
    
    const points = [];
    for (let i = 0; i < xVals.length; i++) {
        // Deflection w is typically in millimeters (mm).
        // Magnification factor of 5000: 1mm -> 5.0 units downward displacement.
        const dy = (dVals[i] * 5.0) / 1000.0; // convert mm to coordinate displacement
        points.push(new THREE.Vector3(xVals[i], yBaseline + dy, zOffset));
    }

    // Draw lines/cylinders between points
    const segmentGeo = new THREE.CylinderGeometry(0.02, 0.02, 1, 8);
    segmentGeo.rotateZ(Math.PI/2);
    
    const segmentMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4 });

    for (let i = 0; i < points.length - 1; i++) {
        const p1 = points[i];
        const p2 = points[i+1];
        
        const dist = p1.distanceTo(p2);
        const midPoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
        
        const mesh = new THREE.Mesh(segmentGeo, segmentMat);
        mesh.position.copy(midPoint);
        mesh.scale.set(dist, 1, 1);
        
        // Orient cylinder along the vector p1 -> p2
        const direction = new THREE.Vector3().subVectors(p2, p1).normalize();
        const alignAxis = new THREE.Vector3(1, 0, 0); // cylinder length axis is X after rotateZ
        const quaternion = new THREE.Quaternion().setFromUnitVectors(alignAxis, direction);
        mesh.quaternion.copy(quaternion);
        
        deflectionRibbon.add(mesh);
    }
}

// ----------------------------------------------------
// ANIMATION LOOP
// ----------------------------------------------------
let lastTime = 0;
function animate(time) {
    requestAnimationFrame(animate);

    // Update Three.js OrbitControls
    if (controls) controls.update();

    // Playback Timeline Progression
    if (isPlaying && dtData) {
        // Advance timeline based on playback speed (every N frames)
        currentT = (currentT + playbackSpeed) % 960;
        updateDashboard();
    } else if (alertActive && dtData) {
        // Update dashboard to keep alert animations pulsing even when paused
        updateDashboard();
    }

    // Render Three.js Scene
    if (renderer && scene && camera) {
        renderer.render(scene, camera);
    }
}

function updateModalAlertsView() {
    if (!modalAlertsContainer) return;
    if (sessionAlertHistory.length === 0) {
        modalAlertsContainer.innerHTML = '<div class="alert-empty-state">No alerts recorded in this session yet.</div>';
    } else {
        // Show in reverse chronological order (newest first)
        modalAlertsContainer.innerHTML = [...sessionAlertHistory]
            .reverse()
            .map(alert => `<div class="alert-item" style="font-size: 11px; margin-bottom: 6px;">${alert}</div>`)
            .join('');
    }
}

function playAlertSound() {
    try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) return;
        const ctx = new AudioContext();
        
        const osc1 = ctx.createOscillator();
        const osc2 = ctx.createOscillator();
        const gain = ctx.createGain();
        
        osc1.type = 'sawtooth';
        osc1.frequency.setValueAtTime(880, ctx.currentTime); // A5 note
        
        osc2.type = 'sine';
        osc2.frequency.setValueAtTime(440, ctx.currentTime); // A4 note
        
        gain.gain.setValueAtTime(0.12, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);
        
        osc1.connect(gain);
        osc2.connect(gain);
        gain.connect(ctx.destination);
        
        osc1.start();
        osc2.start();
        osc1.stop(ctx.currentTime + 0.4);
        osc2.stop(ctx.currentTime + 0.4);
    } catch (e) {
        console.warn("Audio Context playback blocked or unsupported:", e);
    }
}
