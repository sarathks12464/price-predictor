from django.db import models

# Create your models here.
class register_user(models.Model):
    name=models.CharField(max_length=225)
    email=models.EmailField(unique=True)
    phone = models.IntegerField()
    password=models.CharField(max_length=225)

    from django.db import models


class House(models.Model):
    house_age = models.IntegerField()
    area = models.CharField(max_length=225)
    balcony = models.IntegerField()
    bathrooms = models.IntegerField()
    hospital_distance = models.FloatField()
    restaurant_distance = models.FloatField()
    amenity = models.CharField(max_length=500)
    school_distance = models.FloatField()
    shopping_distance = models.FloatField()

    def __str__(self):
        return f"House with {self.area} sqm area"

class Construction(models.Model):
    name = models.CharField(max_length=255)
    address = models.CharField(max_length=500)
    phone_no = models.CharField(max_length=15)
    email = models.EmailField()
    website = models.URLField(blank=True, null=True)
    established_date = models.DateField()
    license_no = models.CharField(max_length=100, unique=True)
    password=models.CharField(max_length=10)

    def __str__(self):
        return self.name

class Upldprojects(models.Model):
    company=models.ForeignKey(Construction,on_delete=models.CASCADE)
    project_description=models.CharField(max_length=225)
    project_cost=models.CharField(max_length=25)
    project_duration=models.CharField(max_length=25)
    project_image=models.ImageField(upload_to='images/')
    created_at=models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.project_name



class Designer(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=15, blank=True, null=True)
    date_of_birth = models.DateField(null=True, blank=True)
    location = models.CharField(max_length=100, blank=True, null=True)  
    gender = models.CharField(max_length=10, choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')], blank=True, null=True)
    company_name = models.CharField(max_length=255, blank=True, null=True)
    company_address = models.TextField(blank=True, null=True)
    profession = models.CharField(max_length=100, choices=[('Interior', 'Interior Design'), ('Exterior', 'Exterior Design')], blank=True, null=True)
    years_of_experience = models.IntegerField(default=0) 
    certifications = models.TextField(blank=True, null=True)  
    skills = models.TextField(blank=True, null=True) 
    password = models.CharField(max_length=225)
    profile_picture = models.ImageField(upload_to='designer_profile_pics/', blank=True, null=True)
    website = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name



class Design(models.Model):
    designer = models.ForeignKey(Designer, on_delete=models.CASCADE, related_name='designs')  
    title = models.CharField(max_length=255) 
    description = models.TextField(blank=True, null=True) 
    design_file = models.FileField(upload_to='designer_designs/', blank=True, null=True)
    image_preview = models.ImageField(upload_to='designer_design_previews/', blank=True, null=True)  
    cost = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)  
    created_at = models.DateTimeField(auto_now_add=True)  
    updated_at = models.DateTimeField(auto_now=True) 

    def __str__(self):
        return self.title
