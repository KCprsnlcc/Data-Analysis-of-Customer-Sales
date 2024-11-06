# analysis/forms.py

from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField(label='Select a CSV file')

    def clean_file(self):
        file = self.cleaned_data.get('file', False)
        if file:
            if not file.name.endswith('.csv'):
                raise forms.ValidationError("Only CSV files are allowed.")
            if file.size > 5*1024*1024:
                raise forms.ValidationError("File size must be under 5MB.")
            return file
        else:
            raise forms.ValidationError("Couldn't read uploaded file.")
