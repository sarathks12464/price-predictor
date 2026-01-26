from django.contrib import admin
from .models import register_user, House, Construction, Upldprojects, Designer, Design

# Register the register_user model
admin.site.register(register_user)

# Register the House model
admin.site.register(House)

# Register the Construction model
admin.site.register(Construction)

# Register the Upldprojects model
admin.site.register(Upldprojects)

# Register the Designer model
admin.site.register(Designer)

# Register the Design model
admin.site.register(Design)
