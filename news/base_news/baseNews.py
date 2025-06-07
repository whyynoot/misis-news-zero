from rest_framework import serializers

class ClassificationPairSerializer(serializers.Serializer):
    """Serializer for an individual classification pair."""
    class1 = serializers.CharField(max_length=50)
    class2 = serializers.CharField(max_length=50)

class BaseClassificationSerializer(serializers.Serializer):
    """Serializer for validating and creating a classification task."""
    pairs = serializers.ListField(
        child=ClassificationPairSerializer(
        ),
        min_length=1,  # Require at least two pairs
        error_messages={
            'min_length': 'At least two classification pairs are required.'
        }
    )

    def validate_pairs(self, value):
        """Custom validation for classification pairs."""
        if not value:
            raise serializers.ValidationError('The classification pairs list cannot be empty.')

        for pair in value:
            if not pair.get('class1') or not pair.get('class2'):
                raise serializers.ValidationError('Each pair must contain two non-empty class names.')

        return value