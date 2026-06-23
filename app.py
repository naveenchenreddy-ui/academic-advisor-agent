"""
Flask web server for Academic Advisor Agent
Wraps the Python CLI backend with HTTP API
"""
from flask import Flask, render_template, request, jsonify, redirect, session
from flask_cors import CORS
from academic_advisor_agent import AcademicAdvisorAgent
import os
from functools import wraps

app = Flask(__name__)
app.secret_key = 'academic-advisor-secret-key'
CORS(app)

# Initialize the agent once
agent = AcademicAdvisorAgent()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'student_id' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'student_id' in session:
        return redirect('/dashboard')
    return redirect('/login')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """Handle chat messages from frontend"""
    try:
        data = request.json
        student_id = session.get('student_id', 'guest')
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get response from advisor
        response = agent.advise(message, student_id=student_id)
        
        return jsonify({
            'success': True,
            'answer': response['answer'],
            'role': response['role'],
            'role_emoji': response['role_emoji'],
            'sources': response['sources']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/profile', methods=['GET'])
@login_required
def get_profile():
    """Get student profile"""
    try:
        student_id = session.get('student_id', 'guest')
        if student_id in agent.profiles:
            profile = agent.profiles[student_id]
            return jsonify({
                'name': profile.name,
                'context': profile.context_string(),
                'interactions': len(profile.session_log),
                'course_count': len(profile.courses),
                'total_credits': profile.total_credits() if hasattr(profile, 'total_credits') else 0,
            })
        return jsonify({'error': 'Profile not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/courses', methods=['GET'])
@login_required
def get_courses():
    """List enrolled courses and credit totals."""
    try:
        student_id = session.get('student_id', 'guest')
        summary = agent.course_summary(student_id)
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/courses', methods=['POST'])
@login_required
def add_course():
    """Add a new course to student profile."""
    try:
        student_id = session.get('student_id', 'guest')
        data = request.json or {}
        course = {
            'name': data.get('name', '').strip(),
            'code': data.get('code', '').strip().upper(),
            'credits': data.get('credits', 0),
            'type': data.get('type', 'core').strip().lower(),
        }
        if not course['name'] or not course['code']:
            return jsonify({'error': 'Course name and code are required'}), 400
        if isinstance(course['credits'], str) and course['credits'].isdigit():
            course['credits'] = int(course['credits'])

        courses = agent.add_course(student_id, course)
        summary = agent.course_summary(student_id)
        return jsonify({'success': True, 'courses': courses, **summary})
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/courses/<course_code>', methods=['DELETE'])
@login_required
def delete_course(course_code):
    """Remove a course from student profile."""
    try:
        student_id = session.get('student_id', 'guest')
        agent.remove_course(student_id, course_code)
        summary = agent.course_summary(student_id)
        return jsonify({'success': True, **summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/report', methods=['GET'])
@login_required
def get_report():
    """Generate progress report"""
    try:
        student_id = session.get('student_id', 'guest')
        report = agent.generate_progress_report(student_id)
        return jsonify({'report': report})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system stats"""
    try:
        stats = agent.stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    """Handle login from frontend"""
    try:
        data = request.json
        student_id = data.get('student_id', '').strip()
        student_name = data.get('student_name', '').strip()
        
        if not student_id:
            return jsonify({'error': 'Student ID required'}), 400
        
        # Store in session
        session['student_id'] = student_id
        session['student_name'] = student_name or f'Student {student_id}'
        
        return jsonify({'success': True, 'redirect': '/dashboard'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """Handle logout"""
    session.clear()
    return jsonify({'success': True, 'redirect': '/login'})

@app.route('/api/student-info', methods=['GET'])
@login_required
def get_student_info():
    """Get current student info"""
    return jsonify({
        'id': session.get('student_id'),
        'name': session.get('student_name', 'Guest')
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

