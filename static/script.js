// JavaScript for Fake News Detection System

document.addEventListener('DOMContentLoaded', function() {
    // Form event listeners
    document.getElementById('textForm').addEventListener('submit', handleTextSubmit);
    document.getElementById('urlForm').addEventListener('submit', handleUrlSubmit);
});

function handleTextSubmit(e) {
    e.preventDefault();
    
    const text = document.getElementById('newsText').value.trim();
    const source = document.getElementById('newsSource').value.trim() || 'User Input';
    
    if (!text) {
        alert('Please enter some text to analyze.');
        return;
    }
    
    analyzeContent('text', text, source);
}

function handleUrlSubmit(e) {
    e.preventDefault();
    
    const url = document.getElementById('newsUrl').value.trim();
    
    if (!url) {
        alert('Please enter a URL to analyze.');
        return;
    }
    
    if (!isValidUrl(url)) {
        alert('Please enter a valid URL (including http:// or https://).');
        return;
    }
    
    analyzeContent('url', url);
}

function quickAnalyze() {
    const text = document.getElementById('newsText').value.trim();
    
    if (!text) {
        alert('Please enter some text to analyze.');
        return;
    }
    
    showLoading();
    
    fetch('/quick-analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            content: text
        })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.error) {
            showError(data.error);
        } else {
            displayQuickResults(data);
        }
    })
    .catch(error => {
        hideLoading();
        showError('Quick analysis failed: ' + error.message);
    });
}

function analyzeContent(type, content, source = null) {
    showLoading();
    
    const payload = {
        type: type,
        content: content
    };
    
    if (source) {
        payload.source = source;
    }
    
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data);
        }
    })
    .catch(error => {
        hideLoading();
        showError('Analysis failed: ' + error.message);
    });
}

function displayResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    
    // Article info
    displayArticleInfo(data.article);
    
    // Main result
    displayMainResult(data);
    
    // Confidence scores
    displayConfidenceScores(data);
    
    // Feature analysis
    displayFeatureAnalysis(data.features_summary);
    
    // Interpretation
    displayInterpretation(data.interpretation);
    
    // Processing info
    displayProcessingInfo(data);
    
    // Show results with animation
    resultsSection.style.display = 'block';
    resultsSection.classList.add('slide-in');
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function displayQuickResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    
    // Simple article info for quick mode
    document.getElementById('articleInfo').innerHTML = `
        <div class="article-meta">
            <h5><i class="fas fa-bolt me-2"></i>Quick Analysis Mode</h5>
            <span class="badge bg-info">Fast Processing</span>
            <span class="badge bg-secondary">Basic Features</span>
        </div>
    `;
    
    // Main result
    displayMainResult(data);
    
    // Confidence scores
    displayConfidenceScores(data);
    
    // Simple feature analysis
    document.getElementById('featureAnalysis').innerHTML = `
        <div class="feature-item">
            <i class="fas fa-info-circle me-2"></i>
            Quick mode uses basic TF-IDF features for fast analysis
        </div>
    `;
    
    // Simple interpretation
    const prediction = data.prediction;
    const confidence = data.confidence;
    
    let interpretation = '';
    if (prediction === 'FAKE') {
        if (confidence > 0.7) {
            interpretation = '⚠️ High likelihood of fake news detected.';
        } else {
            interpretation = '🔍 Some indicators suggest this might be fake news.';
        }
    } else {
        if (confidence > 0.7) {
            interpretation = '✅ Content appears to be legitimate news.';
        } else {
            interpretation = '🔍 Content appears legitimate but verify with other sources.';
        }
    }
    
    document.getElementById('interpretation').innerHTML = `
        <div class="interpretation-item">
            <p class="mb-0">${interpretation}</p>
        </div>
    `;
    
    // Processing info
    document.getElementById('processingInfo').innerHTML = `
        <i class="fas fa-clock me-1"></i>Processed in ${data.processing_time}s (Quick Mode)
    `;
    
    // Show results
    resultsSection.style.display = 'block';
    resultsSection.classList.add('slide-in');
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function displayArticleInfo(article) {
    let html = `
        <div class="article-meta">
            <h5><i class="fas fa-newspaper me-2"></i>${article.title}</h5>
            <div class="mb-2">
                <span class="badge bg-primary">${article.source}</span>
                <span class="badge bg-secondary">${article.word_count} words</span>
    `;
    
    if (article.url) {
        html += `<a href="${article.url}" target="_blank" class="badge bg-info text-decoration-none">
                    <i class="fas fa-external-link-alt me-1"></i>View Original
                 </a>`;
    }
    
    if (article.authors && article.authors.length > 0) {
        html += `<span class="badge bg-success">By: ${article.authors.join(', ')}</span>`;
    }
    
    if (article.publish_date) {
        html += `<span class="badge bg-warning text-dark">${article.publish_date}</span>`;
    }
    
    html += `
            </div>
        </div>
    `;
    
    document.getElementById('articleInfo').innerHTML = html;
}

function displayMainResult(data) {
    const prediction = data.prediction;
    const confidence = data.confidence;
    const riskLevel = data.risk_level;
    
    const resultClass = prediction === 'FAKE' ? 'result-fake' : 'result-real';
    const icon = prediction === 'FAKE' ? 'fas fa-exclamation-triangle' : 'fas fa-check-circle';
    const riskClass = getRiskClass(riskLevel);
    
    const html = `
        <div class="${resultClass}">
            <h2><i class="${icon} me-2"></i>${prediction} NEWS</h2>
            <p class="mb-2">Confidence: ${(confidence * 100).toFixed(1)}%</p>
            <span class="${riskClass}">${riskLevel} RISK</span>
        </div>
    `;
    
    document.getElementById('mainResult').innerHTML = html;
}

function displayConfidenceScores(data) {
    const fakeProb = data.fake_probability;
    const realProb = data.real_probability;
    
    const html = `
        <div class="mb-3">
            <div class="d-flex justify-content-between">
                <span><i class="fas fa-times-circle text-danger me-1"></i>Fake News</span>
                <span>${(fakeProb * 100).toFixed(1)}%</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill confidence-fake" style="width: ${fakeProb * 100}%">
                    ${(fakeProb * 100).toFixed(1)}%
                </div>
            </div>
        </div>
        <div class="mb-3">
            <div class="d-flex justify-content-between">
                <span><i class="fas fa-check-circle text-success me-1"></i>Real News</span>
                <span>${(realProb * 100).toFixed(1)}%</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill confidence-real" style="width: ${realProb * 100}%">
                    ${(realProb * 100).toFixed(1)}%
                </div>
            </div>
        </div>
    `;
    
    document.getElementById('confidenceScores').innerHTML = html;
}

function displayFeatureAnalysis(features) {
    let html = '';
    
    for (const [key, value] of Object.entries(features)) {
        const featureClass = getFeatureClass(key, value);
        const icon = getFeatureIcon(key);
        const displayName = key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
        
        html += `
            <div class="feature-item ${featureClass}">
                <i class="${icon} me-2"></i>
                <strong>${displayName}:</strong> ${value}
            </div>
        `;
    }
    
    if (!html) {
        html = '<div class="feature-item">No detailed features available</div>';
    }
    
    document.getElementById('featureAnalysis').innerHTML = html;
}

function displayInterpretation(interpretations) {
    let html = '';
    
    interpretations.forEach(interpretation => {
        html += `
            <div class="interpretation-item">
                <p class="mb-0">${interpretation}</p>
            </div>
        `;
    });
    
    document.getElementById('interpretation').innerHTML = html;
}

function displayProcessingInfo(data) {
    const html = `
        <i class="fas fa-clock me-1"></i>Processed in ${data.processing_time}s using advanced AI analysis
        <i class="fas fa-brain ms-3 me-1"></i>714 features analyzed
        <i class="fas fa-cogs ms-3 me-1"></i>Ensemble model
    `;
    
    document.getElementById('processingInfo').innerHTML = html;
}

function showLoading() {
    document.getElementById('loadingIndicator').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
}

function hideLoading() {
    document.getElementById('loadingIndicator').style.display = 'none';
}

function showError(message) {
    alert('Error: ' + message);
}

function loadExample(element) {
    const text = element.textContent.replace(/"/g, '');
    document.getElementById('newsText').value = text;
    
    // Switch to text tab if not already active
    const textTab = document.getElementById('text-tab');
    const textTabPane = document.getElementById('text-input');
    
    textTab.classList.add('active');
    textTabPane.classList.add('show', 'active');
    
    // Remove active from URL tab
    document.getElementById('url-tab').classList.remove('active');
    document.getElementById('url-input').classList.remove('show', 'active');
    
    // Scroll to input
    document.getElementById('newsText').scrollIntoView({ behavior: 'smooth' });
    document.getElementById('newsText').focus();
}

function isValidUrl(string) {
    try {
        new URL(string);
        return true;
    } catch (_) {
        return false;
    }
}

function getRiskClass(riskLevel) {
    switch (riskLevel.toLowerCase()) {
        case 'low': return 'risk-low';
        case 'medium': return 'risk-medium';
        case 'high': return 'risk-high';
        case 'very high': return 'risk-very-high';
        default: return 'risk-medium';
    }
}

function getFeatureClass(key, value) {
    if (key === 'source_type') {
        if (value === 'Credible') return 'feature-positive';
        if (value === 'Suspicious') return 'feature-negative';
    }
    
    if (key === 'sentiment') {
        if (value === 'Positive') return 'feature-positive';
        if (value === 'Negative') return 'feature-negative';
    }
    
    if (key === 'writing_style') {
        if (value.includes('Sensational') || value.includes('Aggressive')) {
            return 'feature-negative';
        }
    }
    
    return 'feature-neutral';
}

function getFeatureIcon(key) {
    switch (key) {
        case 'sentiment': return 'fas fa-heart';
        case 'source_type': return 'fas fa-building';
        case 'writing_style': return 'fas fa-pen';
        default: return 'fas fa-info-circle';
    }
}
