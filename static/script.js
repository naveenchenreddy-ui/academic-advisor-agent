// Global state
let currentStudent = {
    id: 'guest',
    name: 'Guest'
};
let currentCourses = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadStudentInfo();
    setupEventListeners();
    loadCourses();
});

function setupEventListeners() {
    // Logout
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }

    // Navigation
    document.querySelectorAll('.menu-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const page = item.dataset.page;
            navigateTo(page);
        });
    });

    // Show/hide add course form
    const showCourseFormBtn = document.getElementById('showCourseFormBtn');
    if (showCourseFormBtn) {
        showCourseFormBtn.addEventListener('click', toggleCourseForm);
    }

    const cancelCourseBtn = document.getElementById('cancelCourseBtn');
    if (cancelCourseBtn) {
        cancelCourseBtn.addEventListener('click', closeCourseForm);
    }

    const newCourseForm = document.getElementById('newCourseForm');
    if (newCourseForm) {
        newCourseForm.addEventListener('submit', handleAddCourseForm);
    }

    // Chat
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
}

function loadStudentInfo() {
    fetch('/api/student-info')
        .then(res => res.json())
        .then(data => {
            currentStudent.id = data.id;
            currentStudent.name = data.name;
            const userInfo = document.getElementById('userInfo');
            if (userInfo) {
                userInfo.textContent = currentStudent.name;
            }
        })
        .catch(err => console.error('Error loading student info:', err));
}

function handleLogout() {
    fetch('/api/logout', { method: 'POST' })
        .then(() => {
            window.location.href = '/login';
        })
        .catch(err => console.error('Logout error:', err));
}

function toggleCourseForm() {
    const courseForm = document.getElementById('courseForm');
    if (!courseForm) return;

    courseForm.classList.toggle('hidden');
    if (!courseForm.classList.contains('hidden')) {
        document.getElementById('courseName').focus();
    }
}

function closeCourseForm() {
    const courseForm = document.getElementById('courseForm');
    if (!courseForm) return;
    courseForm.classList.add('hidden');
    document.getElementById('newCourseForm').reset();
}

function handleAddCourseForm(event) {
    event.preventDefault();

    const name = document.getElementById('courseName').value.trim();
    const code = document.getElementById('courseCode').value.trim().toUpperCase();
    const credits = parseInt(document.getElementById('courseCredits').value, 10);
    const type = document.getElementById('courseType').value;

    if (!name || !code || Number.isNaN(credits)) {
        alert('Please fill in course name, code, and credits.');
        return;
    }

    fetch('/api/courses', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            name,
            code,
            credits,
            type
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            currentCourses = data.courses;
            renderCoursePage(data);
            closeCourseForm();
        } else {
            alert(data.error || 'Unable to save course.');
        }
    })
    .catch(err => {
        console.error('Error saving course:', err);
        alert('Unable to save course. Please try again.');
    });
}

function loadCourses() {
    fetch('/api/courses')
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                console.error('Error loading courses:', data.error);
                return;
            }
            currentCourses = data.courses || [];
            renderCoursePage(data);
        })
        .catch(err => console.error('Error loading courses:', err));
}

function navigateTo(pageId) {
    document.querySelectorAll('.page-content').forEach(p => {
        p.classList.remove('active');
    });

    const page = document.getElementById(pageId);
    if (page) {
        page.classList.add('active');
    }

    document.querySelectorAll('.menu-item').forEach(item => {
        item.classList.remove('active');
    });
    const activeMenuItem = document.querySelector(`[data-page="${pageId}"]`);
    if (activeMenuItem) {
        activeMenuItem.classList.add('active');
    }

    if (pageId === 'courses') {
        loadCourses();
    }
}

function renderCoursePage(summary) {
    document.getElementById('courseCount').textContent = summary.enrolled_count || 0;
    document.getElementById('totalCredits').textContent = summary.total_credits || 0;
    document.getElementById('electivesCount').textContent = summary.electives_count || 0;
    renderCourseList(currentCourses);
}

function renderCourseList(courses) {
    const courseList = document.getElementById('courseList');
    const placeholder = document.getElementById('coursesPlaceholder');
    if (!courseList || !placeholder) return;

    courseList.innerHTML = '';

    if (!courses || courses.length === 0) {
        placeholder.style.display = 'block';
        return;
    }

    placeholder.style.display = 'none';

    courses.forEach(course => {
        const card = document.createElement('div');
        card.className = 'course-item card';

        const label = document.createElement('div');
        label.className = 'course-label';
        label.innerHTML = `<strong>${escapeHtml(course.name)}</strong><span>${escapeHtml(course.code)}</span>`;

        const meta = document.createElement('div');
        meta.className = 'course-meta';
        meta.innerHTML = `<span>${escapeHtml(course.credits || 0)} credits</span><span class="course-badge">${escapeHtml(course.type || 'core')}</span>`;

        const removeBtn = document.createElement('button');
        removeBtn.className = 'btn-secondary btn-remove';
        removeBtn.textContent = 'Remove';
        removeBtn.addEventListener('click', () => removeCourse(course.code));

        card.appendChild(label);
        card.appendChild(meta);
        card.appendChild(removeBtn);
        courseList.appendChild(card);
    });
}

function removeCourse(courseCode) {
    fetch(`/api/courses/${encodeURIComponent(courseCode)}`, { method: 'DELETE' })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                currentCourses = data.courses || [];
                renderCoursePage(data);
            } else {
                alert(data.error || 'Unable to remove course.');
            }
        })
        .catch(err => {
            console.error('Error removing course:', err);
            alert('Unable to remove course. Please try again.');
        });
}

// Chat
function sendMessage(text = null) {
    const input = document.getElementById('messageInput');
    const message = text || input.value.trim();

    if (!message) return;

    input.value = '';
    addMessageToChat(message, 'user');

    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            message: message
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            addMessageToChat(data.answer, 'advisor');
        } else {
            addMessageToChat('Sorry, I encountered an error: ' + (data.error || 'Unknown error'), 'advisor');
        }
    })
    .catch(err => {
        console.error('Error:', err);
        addMessageToChat('Connection error. Please try again.', 'advisor');
    });
}

function addMessageToChat(text, sender) {
    const messagesDiv = document.getElementById('chatMessages');
    if (!messagesDiv) return;

    const messageEl = document.createElement('div');
    messageEl.className = `message ${sender}-message`;

    if (sender === 'advisor') {
        messageEl.innerHTML = `<strong>Advisor:</strong><p>${escapeHtml(text)}</p>`;
    } else {
        messageEl.innerHTML = `<strong>You:</strong><p>${escapeHtml(text)}</p>`;
    }

    messagesDiv.appendChild(messageEl);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

