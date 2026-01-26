from django.shortcuts import render
from .import models
from django.shortcuts import HttpResponse,redirect
from django.shortcuts import get_object_or_404
from .models import Construction, Upldprojects
# Create your views here.
def index(request):
    return render(request,'index.html')


from django.shortcuts import render, HttpResponse, redirect
from . import models
def home(request):
    return render(request,'home.html')

def register(request):
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        password = request.POST.get('password')
        cpassword = request.POST.get('cpassword')

        if models.register_user.objects.filter(email=email).exists():
            alert = "<script>alert('Email already exists');window.location.href = '/register/';</script>"
            return HttpResponse(alert)

        if password != cpassword:
            alert = "<script>alert('Password Not Match');window.location.href = '/register/';</script>"
            return HttpResponse(alert)

        try:
 
            newusr = models.register_user(name=name, email=email, phone=phone, password=password)
            newusr.save()
            return redirect('login')  
        except Exception as e:
            print(e)
            alert = "<script>alert('An error occurred during registration');window.location.href = '/register/';</script>"
            return HttpResponse(alert)
    else:

        return render(request, 'register.html')
    
def login(request):
    if request.method == 'POST':
        email=request.POST.get('email')
        password=request.POST.get('password')
        try:
            newusr=models.register_user.objects.get(email=email,password=password)
        except:
            alert = "<script>alert('invalid email password');window.location.href = '/login/';</script>"
            return HttpResponse(alert)
        if newusr:
            request.session['email']=email
            return redirect('home')
    else:
        return render(request,'login.html')
def profile_user(request):
    print('ttr')
    if 'email' in request.session:
        email = request.session['email']
        print('j')
        try:
            print('p')
            us = models.register_user.objects.get(email=email)
            prf = models.register_user.objects.filter(email=us.email)
            return render(request,'profile_user.html',{'prf':prf})
        except Exception as e:
                print('e',e)
                alert = "<script>alert('User not found. Please log in again.');window.location.href = '/login/';</script>"
                return HttpResponse(alert)
    else:
        return redirect('/login/')
    
def update_profile(request):
    if 'email' in request.session:
        email = request.session['email']
        try:
            user = models.register_user.objects.get(email=email)
            if request.method == 'POST':
                user.name = request.POST.get('name')
                user.phone = request.POST.get('phone')
                user.password = request.POST.get('password')
                user.save()
                return redirect('/profile_user/')
            return render(request, 'update_profile.html', {'user': user})
        except models.register_user.DoesNotExist:
            alert = "<script>alert('User not found. Please log in again.');window.location.href = '/login/';</script>"
            return HttpResponse(alert)
    else:
        return redirect('/login/')
    
from django.shortcuts import redirect
from django.contrib.auth import logout
import joblib
import os
from django.conf import settings
def user_logout(request):
    logout(request)
    return redirect('index')  



from django.shortcuts import render, redirect
from django.conf import settings
import os
import joblib
from . import models
from django.contrib import messages

# Global variables for models
scaler = None
pca = None
gb_model = None

def load_models():
    """Load all models and return their status"""
    global scaler, pca, gb_model
    
    try:
        sc_path = os.path.join(settings.BASE_DIR, 'models', 'scaler_rt.pkl')
        pca_path = os.path.join(settings.BASE_DIR, 'models', 'pca_rt.pkl')
        gb_model_path = os.path.join(settings.BASE_DIR, 'models', 'gradient_boosting_model_ut.pkl')
        
        scaler = joblib.load(sc_path)
        pca = joblib.load(pca_path)
        gb_model = joblib.load(gb_model_path)
        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

def add_house(request):
    # Load models if they haven't been loaded
    if scaler is None or pca is None or gb_model is None:
        models_loaded = load_models()
        if not models_loaded:
            error_message = "Unable to load prediction models. Please contact administrator."
            return render(request, 'add_house.html', {'error': error_message})

    if request.method == 'POST':
        try:
            # Get form data and convert to appropriate types
            features = [
                float(request.POST.get('house_age', 0)),
                float(request.POST.get('area', 0)),
                float(request.POST.get('balcony', 0)),
                float(request.POST.get('bathrooms', 0)),
                float(request.POST.get('hospital_distance', 0)),
                float(request.POST.get('restaurant_distance', 0)),
                float(request.POST.get('amenity', 0)),
                float(request.POST.get('school_distance', 0)),
                float(request.POST.get('shopping_distance', 0))
            ]

            # Validate input values
            if any(x < 0 for x in features):
                raise ValueError("Negative values are not allowed")

            # Save to database
            new_house = models.House(
                house_age=features[0],
                area=features[1],
                balcony=features[2],
                bathrooms=features[3],
                hospital_distance=features[4],
                restaurant_distance=features[5],
                amenity=features[6],
                school_distance=features[7],
                shopping_distance=features[8]
            )
            new_house.save()

            try:
                # Make prediction
                features_scaled = scaler.transform([features])
                features_pca = pca.transform(features_scaled)
                gb_price = gb_model.predict(features_pca)[0]
                
                # Round the prediction to 2 decimal places
                predicted_price = round(gb_price, 2)
                print("pred", predicted_price)
                
                context = {
                    'house': new_house,
                    'predicted_price': predicted_price,
                    'success': True
                }
            except Exception as pred_error:
                print(f"Prediction error: {str(pred_error)}")
                context = {
                    'house': new_house,
                    'error': "Error making prediction. Please try again.",
                    'success': False
                }

            return render(request, 'add_house.html', context)

        except ValueError as e:
            error_message = "Please enter valid numeric values for all fields."
            return render(request, 'add_house.html', {'error': error_message})
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render(request, 'add_house.html', {'error': error_message})

    return render(request, 'add_house.html')

import joblib
from django.shortcuts import render
from django.http import JsonResponse
import numpy as np

# Load the Gradient Boosting model and scaler
import os
import numpy as np
import joblib
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# Load pre-trained models
gb_modelu_path = os.path.join(settings.BASE_DIR, 'models', 'gradient_boosting_loan_model_new.pkl')
scaler_path = os.path.join(settings.BASE_DIR, 'models', 'scaler_loan_new.pkl')
pcal_path = os.path.join(settings.BASE_DIR, 'models', 'pca_loan_new.pkl')

gb_modelu = joblib.load(gb_modelu_path)
scaler_n = joblib.load(scaler_path)
pcal = joblib.load(pcal_path)

# Define global percentiles based on past data (these should be precomputed and stored)
# Alternatively, load `X` dynamically from a CSV or database if needed.
thresholds_path = os.path.join(settings.BASE_DIR, 'models', 'loan_thresholds.npy')

if os.path.exists(thresholds_path):
    thresholds = np.load(thresholds_path, allow_pickle=True).item()
else:
    thresholds = {
        "high_income_threshold": 80000,  # Example value, adjust based on your data
        "high_savings_threshold": 50000,
        "high_credit_score_threshold": 750
    }

@csrf_exempt
def loan_prediction_view(request):
    if request.method == 'POST':
        try:
            # Get user input
            income = float(request.POST.get('income'))
            credit_score = float(request.POST.get('credit_score'))
            loan_amount = float(request.POST.get('loan_amount'))
            loan_term = float(request.POST.get('loan_term'))
            property_ownership = int(request.POST.get('property_ownership'))
            employment_status = int(request.POST.get('employment_status'))
            monthly_debt = float(request.POST.get('monthly_debt'))
            savings = float(request.POST.get('savings'))

            # 🚨 Strict Rejection Rules 🚨  
            if credit_score < 300:
                return JsonResponse({'result': "Not Approved (Very Low Credit Score)"})
            if loan_amount > income * 50 :
                return JsonResponse({'result': "Not Approved"})
            debt_to_income_ratio = monthly_debt / income
            if debt_to_income_ratio > 0.5:
                return JsonResponse({'result': "Not Approved (High Debt-to-Income Ratio)"})
            
            
            expected_credit_score = 300 + (income / 10000) * 50 - (debt_to_income_ratio * 200) + (savings / 1000) * 5
            print("Credit Score1", expected_credit_score)
            expected_credit_score = max(100, min(expected_credit_score, 850))
            if expected_credit_score < 300:
                return JsonResponse({'result': "Not Approved (Invalid Credit Score)"})
            
            if abs(credit_score - expected_credit_score) > 100:
                credit_score = expected_credit_score
                print("Credit score", credit_score)

            # Scale and transform input
            features = [income, credit_score, loan_amount, loan_term, property_ownership, employment_status, monthly_debt, savings]
            features_scaled = scaler_n.transform([features])
            feature_pca = pcal.transform(features_scaled)

            # Model prediction
            prediction = gb_modelu.predict(feature_pca)[0]

            # Adjust threshold logic
            base_threshold = 0.7
            if credit_score < 600:
                base_threshold += 0.1
            if savings > 50000:
                base_threshold -= 0.1
            if loan_term <= 6:
                base_threshold += 0.15  # Penalizing very short loans
            if loan_amount > income * 5:
                base_threshold += 0.2
            if loan_amount > income * 10:
                base_threshold += 0.3

            # Final decision
            if (credit_score > 750 and property_ownership == 1) or (income > 80000 and savings > 50000):
                result = "Approved"
            else:
                result = "Approved" if prediction < base_threshold else "Not Approved"

            return JsonResponse({'result': result,'adjusted_credit_score': round(credit_score, 2)})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return render(request, 'loan_prediction_form.html')





#CONSTRUCTION COMPANIES

def consthome(request):
    return render(request,'consthome.html')

def constreg(request):
    if request.method=='POST':
        name=request.POST.get('name')
        address=request.POST.get('address')
        phone_no=request.POST.get('phone_no')
        email=request.POST.get('email')
        website=request.POST.get('website')
        established_date=request.POST.get('established_date')
        license_no=request.POST.get('license_no')
        password=request.POST.get('password')
        
        if models.Construction.objects.filter(license_no=license_no).exists():
            alert = "<script>alert('Verify your license number');window.location.href = '/constreg/';</script>"
            return HttpResponse(alert)

        try:
 
            con = models.Construction(name=name, address=address, phone_no=phone_no, email=email, website=website, established_date=established_date, license_no=license_no, password=password)
            con.save()
            return redirect('constlogin')  
        except Exception as e:
            print(e)
            alert = "<script>alert('An error occurred during registration');window.location.href = '/constreg/';</script>"
            return HttpResponse(alert)
    else:

        return render(request, 'constreg.html')

def constlogin(request):
    if request.method == 'POST':
        license_no=request.POST.get('license_no')
        password=request.POST.get('password')
        try:
            con=models.Construction.objects.get(license_no=license_no,password=password)
        except:
            alert = "<script>alert('invalid credentials');window.location.href = '/constlogin/';</script>"
            return HttpResponse(alert)
        if con:
            request.session['license_no']=license_no
            return redirect('consthome')
    else:
        return render(request,'constlogin.html')


def constprofile(request):
    if 'license_no' in request.session:
        license_no = request.session['license_no']
    else:
        # If 'license_no' is not in session, redirect to login
        return redirect('/constlogin/')
    
    try:
        con = models.Construction.objects.get(license_no=license_no)
        prf = models.Construction.objects.filter(license_no=con.license_no)
        return render(request, 'constprofile.html', {'prf': prf})
    except models.Construction.DoesNotExist:
        alert = "<script>alert('User not found. Please log in again.');window.location.href = '/constlogin/';</script>"
        return HttpResponse(alert)

def constupdateprofile(request):
    if 'license_no' in request.session:
        license_no = request.session['license_no']
        try:
            user = models.Construction.objects.get(license_no=license_no)
            if request.method == 'POST':
                user.name = request.POST.get('name')
                user.address = request.POST.get('address')
                user.phone_no = request.POST.get('phone_no')
                user.email = request.POST.get('email')
                user.website = request.POST.get('website')
                user.established_date = request.POST.get('established_date')
                user.license_no = request.POST.get('license_no')
                user.password = request.POST.get('password')
                user.save()
                return redirect('/constprofile/')
            return render(request, 'constupdateprofile.html', {'user': user})
        except models.Construction.DoesNotExist:
            alert = "<script>alert('User not found. Please log in again.');window.location.href = '/constlogin/';</script>"
            return HttpResponse(alert)
    else:
        return redirect('/constlogin/')

def conlogout(request):
    logout(request)
    return redirect('index')

def constructorlist(request):
    con = models.Construction.objects.all()
    return render(request, 'constructorlist.html', {'con': con})


def project_list(request):
    # Fetch all projects for the currently logged-in company
    company = models.Construction.objects.get(license_no=request.session['license_no'])
    projects = models.Upldprojects.objects.filter(company=company)

    # Handle the form submission for uploading projects
    if request.method == 'POST':
        project_description = request.POST.get('project_description')
        project_cost = request.POST.get('project_cost')
        project_duration = request.POST.get('project_duration')
        project_image = request.FILES.get('project_image')

        if project_description and project_cost and project_duration and project_image:
            project = models.Upldprojects(
                company=company,
                project_description=project_description,
                project_cost=project_cost,
                project_duration=project_duration,
                project_image=project_image
            )
            project.save()
            return redirect('project_list')  # Redirect to the same page to show updated list
        else:
            return render(request, 'project_list.html', {'error': 'All fields are required.', 'projects': projects})

    # Render the project list and the file upload form
    return render(request, 'project_list.html', {'projects': projects})

def view_projects(request, cid):
    constructor = get_object_or_404(Construction, id=cid)
    projects = Upldprojects.objects.filter(company=constructor)
    return render(request, 'view_projects.html', {'constructor': constructor, 'projects': projects})


# views.py
import os
from django.shortcuts import render
from django.http import JsonResponse
from huggingface_hub import InferenceClient
from PIL import Image
from io import BytesIO

# Initialize Hugging Face Client
HUGGING_FACE_TOKEN = os.getenv("HUGGING_KEY")
client = InferenceClient(model="stabilityai/stable-diffusion-3-medium-diffusers", token=HUGGING_FACE_TOKEN)

DESIGN_SUGGESTIONS = [
    "Modern minimalist living room with natural light",
    "Scandinavian style kitchen with wooden elements",
    "Cozy bohemian bedroom with plants",
    "Industrial style home office with exposed brick",
    "Contemporary bathroom with marble accents"
]

import os
from django.shortcuts import render
from django.http import JsonResponse
from huggingface_hub import InferenceClient
from PIL import Image
from io import BytesIO
from django.conf import settings

# Initialize Hugging Face Client
HUGGING_FACE_TOKEN = os.getenv("HUGGING_KEY")
client = InferenceClient(model="stabilityai/stable-diffusion-3-medium-diffusers", token=HUGGING_FACE_TOKEN)

DESIGN_SUGGESTIONS = [
    "Modern minimalist living room with natural light",
    "Scandinavian style kitchen with wooden elements",
    "Cozy bohemian bedroom with plants",
    "Industrial style home office with exposed brick",
    "Contemporary bathroom with marble accents"
]

def generate_image(request):
    context = {
        'suggestions': DESIGN_SUGGESTIONS
    }
    
    if request.method == "POST":
        prompt = request.POST.get("prompt", "")
        if not prompt:
            return JsonResponse({"error": "Prompt cannot be empty."}, status=400)

        try:
            # Generate the image from the prompt
            image = client.text_to_image(prompt=prompt)

            # Convert image to bytes
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)

            # Create a unique filename with timestamp
            import time
            filename = f"generated_image_{int(time.time())}.png"
            
            # Use os.path.join with forward slashes for the media path
            relative_path = 'generated_images'
            full_path = os.path.join(settings.MEDIA_ROOT, relative_path).replace('\\', '/')
            os.makedirs(full_path, exist_ok=True)
            
            image_path = os.path.join(full_path, filename).replace('\\', '/')
            
            with open(image_path, "wb") as f:
                f.write(buffer.getvalue())

            # Construct the URL using forward slashes
            image_url = f"{settings.MEDIA_URL}{relative_path}/{filename}".replace('\\', '/')
            image_url=image_url
            return JsonResponse({
                "message": "Image generated successfully!", 
                "image_url": image_url
            })
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
            
    return render(request, "interior_design.html", context)



from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Designer
from django.contrib import messages

def designer_register(request):
    if request.method == 'POST':
        name = request.POST['name']
        email = request.POST['email']
        phone = request.POST['phone']
        date_of_birth = request.POST['date_of_birth']
        location = request.POST['location']
        gender = request.POST['gender']
        company_name = request.POST.get('company_name', '')  
        company_address = request.POST.get('company_address', '')  
        profession = request.POST['profession']
        years_of_experience = request.POST['years_of_experience']
        certifications = request.POST.get('certifications', '')  
        skills = request.POST.get('skills', '')
        password = request.POST['password']
        profile_picture=request.FILES['profile_picture']
        website=request.POST['website']

        try:
            designer = Designer.objects.create(
                name=name,
                email=email,
                phone=phone,
                date_of_birth=date_of_birth,
                location=location,
                gender=gender,
                company_name=company_name,
                company_address=company_address,
                profession=profession,
                years_of_experience=years_of_experience,
                certifications=certifications,
                skills=skills,
                password=password,
                profile_picture=profile_picture,
                website=website,
            )

            messages.success(request, "Registration successful.")
            return redirect('designer_login')

        except Exception as e:
            messages.error(request, f"Error occurred during registration: {e}")
            return redirect('designer_register')
    return render(request, 'designer_register.html')

def designer_login(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']
        try:
            designer = Designer.objects.get(email=email,password=password)
            request.session['email']=email
            return redirect('designer_dashboard')  
        except Designer.DoesNotExist:
            messages.error(request, "Invalid email or password.")
            return redirect('designer_login')
    return render(request, 'designer_login.html')


def designer_dashboard(request):
    semail = request.session['email']
    designer = Designer.objects.get(email=semail)
    return render(request, 'designer_dashboard.html', {'designer': designer})

def designer_logout(request):
    request.session.flush()
    return redirect('index')

# views.py

from django.shortcuts import render, redirect
from .models import Designer
from django.contrib import messages

# Display designer profile
def designer_profile(request):
    # Get the email from the session
    semail = request.session.get('email')
    
    if not semail:
        # If there's no email in session, redirect to login
        return redirect('designer_login')
    
    try:
        designer = Designer.objects.get(email=semail)
        return render(request, 'designer_profile.html', {'designer': designer})
    except Designer.DoesNotExist:
        messages.error(request, "Designer not found.")
        return redirect('designer_login')


# Edit designer profile
def designer_editprofile(request):
    semail = request.session.get('email')
    
    if not semail:
        # If there's no email in session, redirect to login
        return redirect('designer_login')
    
    try:
        designer = Designer.objects.get(email=semail)
        
        if request.method == 'POST':
            designer.name = request.POST.get('name')
            designer.phone = request.POST.get('phone')
            designer.date_of_birth = request.POST.get('date_of_birth')
            designer.location = request.POST.get('location')
            designer.gender = request.POST.get('gender')
            designer.company_name = request.POST.get('company_name')
            designer.company_address = request.POST.get('company_address')
            designer.profession = request.POST.get('profession')
            designer.years_of_experience = request.POST.get('years_of_experience')
            designer.certifications = request.POST.get('certifications')
            designer.skills = request.POST.get('skills')
            designer.password = request.POST.get('password')
            designer.website = request.POST.get('website')

            # Handling profile picture upload
            if 'profile_picture' in request.FILES:
                designer.profile_picture = request.FILES['profile_picture']

            designer.save()

            messages.success(request, "Profile updated successfully.")
            return redirect('designer_profile')
        
        return render(request, 'designer_editprofile.html', {'designer': designer})
    
    except Designer.DoesNotExist:
        messages.error(request, "Designer not found.")
        return redirect('designer_login')


from .models import Design
from django.core.exceptions import ValidationError
from django.core.files.storage import FileSystemStorage

def upload_design(request):
    semail = request.session['email']
    designer = Designer.objects.get(email=semail)
    if request.method == 'POST' and request.FILES.get('design_file'):
        title = request.POST.get('title')
        description = request.POST.get('description')
        design_file = request.FILES.get('design_file')
        image_preview = request.FILES.get('image_preview')
        cost = request.POST.get('cost')

        if design_file.size > 5 * 1024 * 1024:  
            messages.error(request, "File size exceeds the 5 MB limit!")
            return redirect('upload_design')

        try:
            design = Design.objects.create(
                designer=designer, 
                title=title,
                description=description,
                design_file=design_file,
                image_preview=image_preview,
                cost=cost
            )

            messages.success(request, "Design uploaded successfully!")
            return redirect('upload_design')

        except ValidationError as e:
            messages.error(request, f"Error uploading design: {e}")
            return redirect('upload_design')

    designs = Design.objects.filter(designer=designer)
    return render(request, 'upload_design.html', {'designs': designs})


from django.shortcuts import render, get_object_or_404
from .models import Designer, Design

def designer_details(request):
    designer = Designer.objects.all()
    context = {
        'designer': designer,
    }
    return render(request, 'designer_details.html', context)

def view_designs(request,designer_id):
    designer=Designer.objects.get(id=designer_id)
    designs=Design.objects.filter(designer=designer)
    return render(request,'view_designs.html',{'designs':designs,'designer':designer})





from django.shortcuts import render
from .models import register_user, House, Designer, Upldprojects


def admin_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        if username == 'admin' and password == 'admin':
            request.session['id'] = username
            return redirect('admin_dashboard') 
        else:
            return HttpResponse("<script>alert('Login Failed.'); window.location='/admin_login/';</script>")
    else:
        return render(request, 'admin_login.html')

def admin_dashboard(request):
    user_count = register_user.objects.count()
    house_count = House.objects.count()
    designer_count = Designer.objects.count()
    project_count = Upldprojects.objects.count()
    return render(request, 'admin_dashboard.html', {
        'user_count': user_count,
        'house_count': house_count,
        'designer_count': designer_count,
        'project_count': project_count
    })

def manage_users(request):
    users = register_user.objects.all()
    return render(request, 'manage_users.html', {'users': users})

def manage_houses(request):
    houses = House.objects.all()
    return render(request, 'manage_houses.html', {'houses': houses})

def manage_construction(request):
    constructions = Construction.objects.all()
    return render(request, 'manage_construction.html', {'constructions': constructions})

def manage_projects(request):
    projects = Upldprojects.objects.all()
    return render(request, 'manage_projects.html', {'projects': projects})

def manage_designers(request):
    designers = Designer.objects.all()
    return render(request, 'manage_designers.html', {'designers': designers})

def manage_designs(request):
    designs = Design.objects.all()
    return render(request, 'manage_designs.html', {'designs': designs})

def admin_logout(request):
    request.session.flush()
    return redirect('index')


def delete_user(request,id):
    user=register_user.objects.get(id=id)
    user.delete()
    return redirect('manage_users')

def delete_construction(request,id):
    construction=Construction.objects.get(id=id)
    construction.delete()
    return redirect('manage_construction')

def delete_designer(request,id):
    designer=Designer.objects.get(id=id)
    designer.delete()
    return redirect('manage_designers')