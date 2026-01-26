from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    #USER
    path('',views.index,name='index'),
    path('register/',views.register,name='register'),
    path('login/',views.login,name='login'),
    path('home/',views.home,name='home'),
    path('profile_user/', views.profile_user, name='profile_user'),
    path('update_profile/', views.update_profile, name='update_profile'),
    path('logout/', views.user_logout, name='logout'),
    path('add_house/', views.add_house, name='add_house'),
    path('constructorlist/', views.constructorlist, name='constructorlist'),
    path('view_projects/<int:cid>/projects/', views.view_projects, name='view_projects'),

    #CONSTRUCTOR
    path('constreg/', views.constreg, name='constreg'),
    path('constlogin/', views.constlogin, name='constlogin'),
    path('consthome/', views.consthome, name='consthome'),
    path('constprofile/', views.constprofile, name='constprofile'),
    path('constupdateprofile/', views.constupdateprofile, name='constupdateprofile'),
    path('conlogout/', views.conlogout, name='conlogout'),
    path('project_list/', views.project_list, name='project_list'),
    path('loan-predict/', views.loan_prediction_view, name='loan_predict'),
    path('generate-image/', views.generate_image, name='generate_image'),


    #DESIGNER
    path('designer_register/', views.designer_register, name='designer_register'),
    path('designer_login/', views.designer_login, name='designer_login'),
    path('designer_dashboard/', views.designer_dashboard, name='designer_dashboard'),
    path('designer_profile/', views.designer_profile, name='designer_profile'),
    path('designer_editprofile/', views.designer_editprofile, name='designer_editprofile'),
    path('designer_logout/', views.designer_logout, name='designer_logout'),
    path('upload_design/', views.upload_design, name='upload_design'),
    path('designer_details/', views.designer_details, name='designer_details'),
    path('view_designs/<int:designer_id>/', views.view_designs, name='view_designs'),


    #ADMIN
    path('admin_login/', views.admin_login, name='admin_login'),
    path('admin_dashboard/',views.admin_dashboard, name='admin_dashboard'),
    path('admin_logout/', views.admin_logout, name='admin_logout'),
    path('manage_users/', views.manage_users, name='manage_users'),
    path('manage_construction/', views.manage_construction, name='manage_construction'),
    path('manage_projects/', views.manage_projects, name='manage_projects'),
    path('manage_designers/', views.manage_designers, name='manage_designers'),
    path('manage_designs/', views.manage_designs, name='manage_designs'),
    path('delete_user/<int:id>/', views.delete_user, name='delete_user'),
    path('delete_construction/<int:id>/', views.delete_construction, name='delete_construction'),
    path('delete_designer/<int:id>/', views.delete_designer, name='delete_designer'),



]