from setuptools import setup

package_name = 'dummy_vlm'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/dummy_vlm.launch']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Devesh Kaushal',
    maintainer_email='deveshkaushal9@gmail.com',
    description='CMU VLN Challenge 2026 AI module',
    license='MIT',
    entry_points={
        'console_scripts': [
            'dummyVLM = dummy_vlm.node:main',
        ],
    },
)
