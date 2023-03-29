import React from 'react';
import { Container } from 'react-bootstrap';
import '../../App.css'

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="footer py-3 bg-dark text-white">
      <Container>
        <p className="text-center mb-0">
          Copyright &copy; {currentYear} AI Watch
        </p>
      </Container>
    </footer>
  );
};

export default Footer;
