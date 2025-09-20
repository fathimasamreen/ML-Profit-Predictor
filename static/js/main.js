document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('predictionForm');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultDiv = document.getElementById('predictionResult');
    const errorDiv = document.getElementById('errorMessage');

    // Handle form submission - use form submission instead of AJAX for simplicity
    if (form) {
        form.addEventListener('submit', function (e) {
            // Let the form submit normally to Flask, but show loading spinner
            if (form.checkValidity()) {
                if (loadingSpinner) {
                    loadingSpinner.classList.remove('d-none');
                }
            }
        });
    }

    // Real-time validation
    const inputs = document.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', function () {
            const val = parseFloat(this.value.replace(/,/g, ''));

            if (this.value === '') {
                this.setCustomValidity('This field is required');
            } else if (isNaN(val) || val < 0) {
                this.setCustomValidity('Value must be a positive number');
            } else {
                this.setCustomValidity('');
            }
        });

        // Remove formatting commas on focus (if any exist)
        input.addEventListener('focus', function () {
            this.value = this.value.replace(/,/g, '');
        });

        // Prevent pasting non-numeric
        input.addEventListener('paste', function (e) {
            const paste = e.clipboardData.getData('text');
            if (isNaN(paste.replace(/,/g, ''))) {
                e.preventDefault();
            }
        });
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Bootstrap tooltips initialization
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    if (typeof bootstrap !== 'undefined') {
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // Auto-scroll to result if it exists (for page refresh with results)
    if (resultDiv && !resultDiv.classList.contains('d-none')) {
        setTimeout(() => {
            resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }

    // Auto-scroll to error if it exists (for page refresh with errors)
    if (errorDiv && !errorDiv.classList.contains('d-none')) {
        setTimeout(() => {
            errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }
});

// Currency formatting utility
function formatCurrency(amount, minimumFractionDigits = 2) {
    return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        minimumFractionDigits: minimumFractionDigits
    }).format(amount);
}